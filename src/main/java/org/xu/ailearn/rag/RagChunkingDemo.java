package org.xu.ailearn.rag;

import cn.hutool.core.io.FileUtil;
import org.xu.ailearn.util.WordToMarkdown;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RAG 分块（Chunking）Java Demo
 *
 * 覆盖要点：
 *  1. 按段落/句子递归切分，表格与代码块整块保留
 *  2. chunk 大小 300~600 token，重叠 10~20%
 *  3. 元数据：来源文件、页码、标题路径、chunk 类型
 *  4. 双粒度：小块（子块）用于检索，大块（父块）用于生成返回
 */
public class RagChunkingDemo {

    // ============================================================
    // 1. 数据模型
    // ============================================================
    enum ChunkType {TEXT, TABLE, CODE}

    /**
     * 最终入库的块
     */
    static class Chunk {
        String id;
        String text;
        String sourceFile;   // 来源文件
        Integer pageNo;      // 页码
        String headingPath;  // 标题路径，如 "名单管理 > 数据表"
        ChunkType type;      // text / table / code
        int tokenCount;
        String parentId;     // 父块 id，单粒度时为 null

        @Override
        public String toString() {
            return String.format("[%s] type=%s tokens=%d page=%s path=%s parent=%s%n%s",
                    id, type, tokenCount, pageNo, headingPath, parentId, text);
        }
    }

    /**
     * 解析后的语义段（段落 / 表格 / 代码块），是分块的最小语义单位来源
     */
    static class Segment {
        final String text;
        final ChunkType type;
        final String headingPath;
        final Integer pageNo;

        Segment(String text, ChunkType type, String headingPath, Integer pageNo) {
            this.text = text;
            this.type = type;
            this.headingPath = headingPath;
            this.pageNo = pageNo;
        }
    }

    /**
     * 原子单元：TEXT 段进一步切成句子；TABLE/CODE 保持整体
     */
    static class Unit {
        final String text;
        final ChunkType type;
        final String headingPath;
        final Integer pageNo;

        Unit(String text, ChunkType type, String headingPath, Integer pageNo) {
            this.text = text;
            this.type = type;
            this.headingPath = headingPath;
            this.pageNo = pageNo;
        }
    }

    // ============================================================
    // 2. Token 估算（生产环境替换为 jtokkit / tiktoken）
    // ============================================================
    static class TokenEstimator {
        static int estimate(String text) {
            if (text == null || text.isEmpty()) return 0;
            int tokens = 0;
            boolean inWord = false;
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                boolean cjk = (c >= 0x4E00 && c <= 0x9FFF)
                        || (c >= 0x3000 && c <= 0x303F)
                        || (c >= 0xFF00 && c <= 0xFFEF);
                if (cjk) {
                    tokens++;          // 中文字符 ≈ 1 token
                    inWord = false;
                } else if (Character.isLetterOrDigit(c)) {
                    if (!inWord) {
                        tokens++;      // 英文单词 ≈ 1~1.3 token，粗估
                        inWord = true;
                    }
                } else {
                    inWord = false;
                }
            }
            return (int) Math.ceil(tokens * 1.1);
        }
    }

    // ============================================================
    // 3. 结构化切段：保留标题层级、页码、表格/代码完整性
    //    约定：\f 为分页符；# 为标题；``` 为代码围栏；|...| 为表格行
    // ============================================================
    static class DocumentSegmenter {
        private static final Pattern HEADING = Pattern.compile("^(#{1,6})\\s+(.+)$");

        List<Segment> segment(String content, String sourceFile) {
            List<Segment> out = new ArrayList<>();
            List<String> headings = new ArrayList<>();   // 标题栈
            String[] pages = content.split("\f", -1);

            for (int p = 0; p < pages.length; p++) {
                int pageNo = p + 1;
                StringBuilder para = new StringBuilder();
                List<String> tableLines = new ArrayList<>();
                StringBuilder code = new StringBuilder();
                boolean inCode = false;

                for (String line : pages[p].split("\n", -1)) {
                    String trimmed = line.trim();

                    // --- 代码围栏 ---
                    if (trimmed.startsWith("```")) {
                        if (!inCode) {
                            flushPara(para, out, headings, pageNo);
                            flushTable(tableLines, out, headings, pageNo);
                            inCode = true;
                        } else {
                            inCode = false;
                            out.add(new Segment(code.toString(), ChunkType.CODE, path(headings), pageNo));
                            code.setLength(0);
                        }
                        continue;
                    }
                    if (inCode) {
                        if (code.length() > 0) code.append('\n');
                        code.append(line);
                        continue;
                    }

                    // --- 标题：维护层级栈 ---
                    Matcher m = HEADING.matcher(trimmed);
                    if (m.matches()) {
                        flushPara(para, out, headings, pageNo);
                        flushTable(tableLines, out, headings, pageNo);
                        int level = m.group(1).length();
                        while (headings.size() >= level) {
                            headings.remove(headings.size() - 1);
                        }
                        headings.add(m.group(2).trim());
                        continue;
                    }

                    // --- 表格行：整表积累，绝不逐行切 ---
                    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
                        flushPara(para, out, headings, pageNo);
                        tableLines.add(trimmed);
                        continue;
                    }

                    // --- 空行：段落边界 ---
                    if (trimmed.isEmpty()) {
                        flushPara(para, out, headings, pageNo);
                        flushTable(tableLines, out, headings, pageNo);
                        continue;
                    }

                    // --- 正文行 ---
                    flushTable(tableLines, out, headings, pageNo);
                    if (para.length() > 0) para.append('\n');
                    para.append(line);
                }

                // 页尾收尾
                if (inCode) {
                    out.add(new Segment(code.toString(), ChunkType.CODE, path(headings), pageNo));
                }
                flushPara(para, out, headings, pageNo);
                flushTable(tableLines, out, headings, pageNo);
            }
            return out;
        }

        private void flushPara(StringBuilder para, List<Segment> out, List<String> headings, int pageNo) {
            if (para.length() > 0) {
                out.add(new Segment(para.toString().trim(), ChunkType.TEXT, path(headings), pageNo));
                para.setLength(0);
            }
        }

        private void flushTable(List<String> tableLines, List<Segment> out, List<String> headings, int pageNo) {
            if (!tableLines.isEmpty()) {
                out.add(new Segment(String.join("\n", tableLines), ChunkType.TABLE, path(headings), pageNo));
                tableLines.clear();
            }
        }

        private String path(List<String> headings) {
            return String.join(" > ", headings);
        }
    }

    // ============================================================
    // 4. 分块器：贪心打包 + 句子级重叠
    // ============================================================
    static class Chunker {
        final int minTokens;      // 建议 300
        final int maxTokens;      // 建议 600
        final int overlapTokens;  // 建议 maxTokens * 0.10 ~ 0.20

        Chunker(int minTokens, int maxTokens, double overlapRatio) {
            this.minTokens = minTokens;
            this.maxTokens = maxTokens;
            this.overlapTokens = (int) Math.round(maxTokens * overlapRatio);
        }

        List<Chunk> chunk(List<Segment> segments, String sourceFile) {
            List<Unit> units = toUnits(segments);
            List<Chunk> result = new ArrayList<>();
            List<Unit> buffer = new ArrayList<>();
            int bufferTokens = 0;

            for (Unit u : units) {
                int t = TokenEstimator.estimate(u.text);

                if (!buffer.isEmpty() && bufferTokens + t > maxTokens) {
                    flush(result, buffer, sourceFile);
                    // 新块带上尾部重叠（只在 TEXT 单元间重叠，表格/代码不参与）
                    buffer = overlapCarry(buffer);
                    bufferTokens = tokenSum(buffer);
                    if (bufferTokens + t > maxTokens) {
                        buffer = new ArrayList<>();  // 重叠都装不下，放弃重叠
                        bufferTokens = 0;
                    }
                }
                buffer.add(u);
                bufferTokens += t;
            }

            // 尾块太小则并入前块，避免碎片
            if (!buffer.isEmpty()) {
                if (bufferTokens < minTokens && !result.isEmpty()
                        && result.get(result.size() - 1).tokenCount + bufferTokens <= maxTokens * 1.5) {
                    mergeTail(result, buffer);
                } else {
                    flush(result, buffer, sourceFile);
                }
            }
            return result;
        }

        // TEXT 段切成句子；TABLE/CODE 保持整体（原子单元，硬切禁区）
        private List<Unit> toUnits(List<Segment> segments) {
            List<Unit> units = new ArrayList<>();
            for (Segment s : segments) {
                if (s.type != ChunkType.TEXT) {
                    units.add(new Unit(s.text, s.type, s.headingPath, s.pageNo));
                } else {
                    for (String sent : splitSentences(s.text)) {
                        units.add(new Unit(sent, ChunkType.TEXT, s.headingPath, s.pageNo));
                    }
                }
            }
            return units;
        }

        private List<String> splitSentences(String text) {
            String[] parts = text.split("(?<=[。！？!?；;])\\s*");
            List<String> out = new ArrayList<>();
            for (String p : parts) {
                if (!p.trim().isEmpty()) out.add(p.trim());
            }
            return out.isEmpty() ? Arrays.asList(text) : out;
        }

        private void flush(List<Chunk> out, List<Unit> buffer, String sourceFile) {
            if (buffer.isEmpty()) return;
            StringBuilder sb = new StringBuilder();
            ChunkType type = buffer.get(0).type;
            boolean uniform = true;
            for (Unit u : buffer) {
                if (sb.length() > 0) sb.append("\n\n");
                sb.append(u.text);
                if (u.type != type) uniform = false;
            }
            Chunk c = new Chunk();
            c.id = sourceFile + "#" + out.size();
            c.text = sb.toString();
            c.sourceFile = sourceFile;
            c.pageNo = buffer.get(0).pageNo;
            c.headingPath = buffer.get(0).headingPath;
            c.type = uniform ? type : ChunkType.TEXT;
            c.tokenCount = TokenEstimator.estimate(c.text);
            out.add(c);
        }

        private List<Unit> overlapCarry(List<Unit> buffer) {
            List<Unit> carried = new ArrayList<>();
            int tokens = 0;
            for (int i = buffer.size() - 1; i >= 0; i--) {
                Unit u = buffer.get(i);
                if (u.type != ChunkType.TEXT) break;   // 不跨表格/代码重叠
                tokens += TokenEstimator.estimate(u.text);
                carried.add(0, u);
                if (tokens >= overlapTokens) break;
            }
            return carried;
        }

        private void mergeTail(List<Chunk> result, List<Unit> buffer) {
            Chunk last = result.get(result.size() - 1);
            StringBuilder sb = new StringBuilder(last.text);
            for (Unit u : buffer) {
                sb.append("\n\n").append(u.text);
                if (u.type != last.type) last.type = ChunkType.TEXT;
            }
            last.text = sb.toString();
            last.tokenCount = TokenEstimator.estimate(last.text);
        }

        private int tokenSum(List<Unit> units) {
            int sum = 0;
            for (Unit u : units) sum += TokenEstimator.estimate(u.text);
            return sum;
        }
    }

    // ============================================================
    // 5. 双粒度：父块（生成返回）+ 子块（检索命中）
    // ============================================================
    static class DualGranularityChunker {
        final Chunker childChunker;
        final int parentMaxTokens;

        DualGranularityChunker(int childMin, int childMax, double overlapRatio, int parentMaxTokens) {
            this.childChunker = new Chunker(childMin, childMax, overlapRatio);
            this.parentMaxTokens = parentMaxTokens;
        }

        static class DualChunks {
            final List<Chunk> parents = new ArrayList<>();
            final List<Chunk> children = new ArrayList<>();
        }

        DualChunks chunk(List<Segment> segments, String sourceFile) {
            DualChunks dc = new DualChunks();
            for (List<Segment> group : packParentGroups(segments)) {
                // 父块：一个标题单元内的完整上下文
                Chunk parent = new Chunk();
                parent.id = sourceFile + "@p" + dc.parents.size();
                parent.sourceFile = sourceFile;
                parent.headingPath = group.get(0).headingPath;
                parent.pageNo = group.get(0).pageNo;
                parent.type = ChunkType.TEXT;
                StringBuilder sb = new StringBuilder();
                for (Segment s : group) {
                    if (sb.length() > 0) sb.append("\n\n");
                    sb.append(s.text);
                }
                parent.text = sb.toString();
                parent.tokenCount = TokenEstimator.estimate(parent.text);
                dc.parents.add(parent);

                // 子块：在父块范围内做小粒度切分，携带 parentId
                List<Chunk> kids = childChunker.chunk(group, sourceFile);
                for (int i = 0; i < kids.size(); i++) {
                    Chunk k = kids.get(i);
                    k.id = parent.id + "-c" + i;
                    k.parentId = parent.id;
                    dc.children.add(k);
                }
            }
            return dc;
        }

        /**
         * 按标题边界 + token 预算把 segments 打包成父块组
         */
        private List<List<Segment>> packParentGroups(List<Segment> segments) {
            List<List<Segment>> groups = new ArrayList<>();
            List<Segment> current = new ArrayList<>();
            int tokens = 0;
            for (Segment s : segments) {
                int t = TokenEstimator.estimate(s.text);
                boolean headingChanged = !current.isEmpty()
                        && !s.headingPath.equals(current.get(current.size() - 1).headingPath);
                if (!current.isEmpty() && (headingChanged || tokens + t > parentMaxTokens)) {
                    groups.add(current);
                    current = new ArrayList<>();
                    tokens = 0;
                }
                current.add(s);
                tokens += t;
            }
            if (!current.isEmpty()) groups.add(current);
            return groups;
        }
    }

    // ============================================================
    // 6. Demo：跑通全流程
    // ============================================================
    public static void main(String[] args) {
        // 约定：\f 为分页符（模拟 PDF/Word 解析后的分页）
        String filePath = "C:\\Users\\92780\\Desktop\\1111.docx";
        String fileName = FileUtil.getName(filePath);  // "反洗钱名单管理办法.docx"
        String doc = WordToMarkdown.convert(filePath);  // Word → markdown 风格文本

        List<Segment> segments = new DocumentSegmenter().segment(doc, fileName);
        System.out.println("===== 切段结果（段落/表格/代码保持完整）=====");
        for (Segment s : segments) {
            System.out.printf("(%s p%d [%s]) %s%n%n", s.type, s.pageNo, s.headingPath, brief(s.text));
        }
        System.out.println("===== 切段结果（段落/表格/代码保持完整）=====");
        for (Segment s : segments) {
            System.out.printf("(%s p%d [%s]) %s%n%n", s.type, s.pageNo, s.headingPath, brief(s.text));
        }
        List<Chunk> chunks = new Chunker(300, 600, 0.15).chunk(segments, fileName);
        for (Chunk c : chunks) {
            System.out.println(c);
            System.out.println("---");
        }
    }

    static String brief(String s) {
        String one = s.replaceAll("\\s+", " ").trim();
        return one.length() > 60 ? one.substring(0, 60) + "..." : one;
    }
}
