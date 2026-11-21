package org.xu.ailearn.util;

import cn.hutool.core.io.FileUtil;
import org.apache.poi.xwpf.usermodel.IBodyElement;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFTable;
import org.apache.poi.xwpf.usermodel.XWPFTableCell;
import org.apache.poi.xwpf.usermodel.XWPFTableRow;

import java.io.InputStream;
import java.util.List;

/**
 * Word → markdown 风格文本，供 DocumentSegmenter 解析
 *
 * 约定：
 *   一级标题 → "# xxx"
 *   二级标题 → "## xxx"
 *   N 级标题 → "#"*N + " xxx"
 *   表格     → "| col1 | col2 |" 格式
 *   段落     → 原文输出
 */
public class WordToMarkdown {

    /**
     * 读取 Word 文件，转成 DocumentSegmenter 能识别的文本
     *
     * @param filePath Word 文件路径（.docx）
     * @return markdown 风格纯文本
     */
    public static String convert(String filePath) {
        InputStream is = null;
        XWPFDocument doc = null;
        try {
            is = FileUtil.getInputStream(filePath);
            doc = new XWPFDocument(is);

            StringBuilder sb = new StringBuilder();
            List<IBodyElement> elements = doc.getBodyElements();

            for (IBodyElement element : elements) {
                // JDK 8 不支持 instanceof 模式匹配，需要手动转型
                if (element instanceof XWPFParagraph) {
                    XWPFParagraph para = (XWPFParagraph) element;
                    appendParagraph(sb, para);
                } else if (element instanceof XWPFTable) {
                    XWPFTable table = (XWPFTable) element;
                    appendTable(sb, table);
                }
            }
            return sb.toString();

        } catch (Exception e) {
            throw new RuntimeException("读取 Word 文件失败: " + filePath, e);
        } finally {
            closeQuietly(doc);
            closeQuietly(is);
        }
    }

    // ---- 段落：根据样式判断是否为标题 ----
    private static void appendParagraph(StringBuilder sb, XWPFParagraph para) {
        String text = para.getText();
        if (text == null || text.trim().isEmpty()) {
            sb.append("\n");
            return;
        }

        String styleName = para.getStyle();
        int level = parseHeadingLevel(styleName, para);

        if (level > 0) {
            sb.append("\n")
                    .append(repeat("#", level))   // JDK 8 没有 String.repeat()
                    .append(" ")
                    .append(text.trim())
                    .append("\n");
        } else {
            sb.append(text.trim()).append("\n\n");
        }
    }

    /**
     * 判断标题层级
     */
    private static int parseHeadingLevel(String styleName, XWPFParagraph para) {
        if (styleName != null) {
            String normalized = styleName.toLowerCase().replace(" ", "");
            if (normalized.contains("heading") || normalized.contains("标题")) {
                for (int i = 1; i <= 9; i++) {
                    if (normalized.endsWith(String.valueOf(i))) {
                        return i;
                    }
                }
            }
        }
        // 兜底：从 XML 中取 outlineLvl
        try {
            // getVal() 返回 BigInteger，必须转成 int 才能做算术比较
            int outlineLvl = para.getCTP().getPPr().getOutlineLvl().getVal().intValue();
            if (outlineLvl >= 0 && outlineLvl < 9) {
                return outlineLvl + 1;
            }
        } catch (Exception ignored) {
        }
        return 0;
    }

    // ---- 表格：转成 | col1 | col2 | 格式 ----
    private static void appendTable(StringBuilder sb, XWPFTable table) {
        sb.append("\n");
        List<XWPFTableRow> rows = table.getRows();
        for (int i = 0; i < rows.size(); i++) {
            XWPFTableRow row = rows.get(i);
            List<XWPFTableCell> cells = row.getTableCells();
            sb.append("| ");
            for (int j = 0; j < cells.size(); j++) {
                sb.append(cells.get(j).getText().trim());
                if (j < cells.size() - 1) {
                    sb.append(" | ");
                }
            }
            sb.append(" |").append("\n");
            // 表头后加分隔行
            if (i == 0) {
                sb.append("|");
                for (int j = 0; j < cells.size(); j++) {
                    sb.append(" --- |");
                }
                sb.append("\n");
            }
        }
        sb.append("\n");
    }

    // ---- JDK 8 兼容工具方法 ----

    /**
     * 替代 JDK 11 的 String.repeat()
     */
    private static String repeat(String str, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) {
            sb.append(str);
        }
        return sb.toString();
    }

    /**
     * 静默关闭，避免 finally 里再套 try-catch
     */
    private static void closeQuietly(java.io.Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception ignored) {
            }
        }
    }
}


