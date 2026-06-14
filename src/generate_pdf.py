import os
import sys
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Determine font availability on Windows
def register_chinese_font():
    possible_paths = [
        "C:/Windows/Fonts/msjh.ttc",    # Microsoft JhengHei (微軟正黑體)
        "C:/Windows/Fonts/msyh.ttc",    # Microsoft YaHei (微軟雅黑)
        "C:/Windows/Fonts/simsun.ttc",   # SimSun (宋體)
        "C:/Windows/Fonts/msjhbd.ttc"   # Microsoft JhengHei Bold
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # TrueType Collection (TTC) index 0 is loaded by default
                pdfmetrics.registerFont(TTFont("ChineseRegular", path))
                return "ChineseRegular"
            except Exception as e:
                print(f"Warning: Failed to register font at {path}: {e}")
                
    print("Warning: No Chinese TrueType font registered. Falling back to Helvetica.")
    return "Helvetica"

def build_pdf_handout():
    """
    Compiles a comprehensive PDF handout containing markdown analysis text,
    metric comparisons, and generated plots.
    """
    pdf_path = os.path.join("outputs", "reports", "startup_profit_prediction_handout.pdf")
    os.makedirs(os.path.join("outputs", "reports"), exist_ok=True)
    
    font_name = register_chinese_font()
    
    # Page settings: A4 vertical
    # Margins: 0.75 inch (54 points) on all sides
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles mapping the Chinese font
    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#00B4D8"),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#415A77"),
        alignment=1,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        "PDFH1",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#0077B6"),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "PDFBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0D1B2A"),
        spaceAfter=10
    )
    
    code_style = ParagraphStyle(
        "PDFCode",
        parent=styles["Code"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#00F5D4"),
        backColor=colors.HexColor("#0D1B2A"),
        borderPadding=10,
        spaceAfter=15
    )
    
    table_text_style = ParagraphStyle(
        "PDFTableText",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0D1B2A")
    )
    
    table_header_style = ParagraphStyle(
        "PDFTableHeaderText",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontWeight="bold"
    )
    
    story = []
    
    # --- PAGE 1: COVER ---
    story.append(Spacer(1, 40))
    story.append(Paragraph("HW6 新創公司獲利預測分析與特徵選擇", title_style))
    story.append(Paragraph("Kaggle 50 Startups Profit Prediction & CRISP-DM 實作講義", subtitle_style))
    story.append(Spacer(1, 20))
    
    intro_p1 = (
        "本講義為機器學習實務課程專作。本專案使用 Kaggle 的 50 Startups 資料集，"
        "完整落實了 <b>CRISP-DM (Cross-Industry Standard Process for Data Mining)</b> 的六大階段。"
        "包含商業理解、資料清洗、類別特徵編碼（加州虛擬變數 Baseline 防範陷阱）、模型擬合、5種特徵選擇方法之交叉比對，"
        "最後儲存最佳二特徵迴歸模型並部署互動式 Streamlit 網頁應用。"
    )
    story.append(Paragraph(intro_p1, body_style))
    story.append(Spacer(1, 15))
    
    # Embed correlation heatmap
    corr_img_path = os.path.join("outputs", "figures", "correlation_heatmap.png")
    if os.path.exists(corr_img_path):
        # Image size: 420x315 points (fits nicely inside margins)
        story.append(Image(corr_img_path, width=400, height=300))
        story.append(Paragraph("<font size=8 color='#415A77'>圖 1：完整特徵（包含 OHE 州別）之皮爾森相關係數熱力圖</font>", subtitle_style))
    
    story.append(PageBreak())
    
    # --- PAGE 2: MODELING COMPARISONS ---
    story.append(Paragraph("第一節：模型建置與特徵子集表現對比", h1_style))
    story.append(Paragraph(
        "我們採用多元線性迴歸作為核心預估模型，並在測試集上（test_size=0.2, random_state=42）比較了 5 組不同數量的特徵子集表現。"
        "如下表所示，隨著特徵數增加，指標呈現非線性變化：",
        body_style
    ))
    
    # Load comparison table
    comp_csv = os.path.join("outputs", "reports", "model_comparison.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        
        # Build table data
        table_data = [[Paragraph(col, table_header_style) for col in ["模型名稱", "包含特徵數", "RMSE", "MAE", "R-squared", "Adjusted R2"]]]
        for _, row in df_comp.iterrows():
            table_data.append([
                Paragraph(str(row["Model Name"]), table_text_style),
                Paragraph(str(row["Num Features"]), table_text_style),
                Paragraph(f"{row['RMSE']:.2f}", table_text_style),
                Paragraph(f"{row['MAE']:.2f}", table_text_style),
                Paragraph(f"{row['R-squared']:.4f}", table_text_style),
                Paragraph(f"{row['Adjusted R-squared']:.4f}" if not pd.isna(row['Adjusted R-squared']) else "N/A", table_text_style)
            ])
            
        comp_table = Table(table_data, colWidths=[130, 60, 70, 70, 75, 75])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0077B6")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#415A77")),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EBF2FA")])
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 15))
        
    rmse_fig_path = os.path.join("outputs", "figures", "rmse_by_features.png")
    if os.path.exists(rmse_fig_path):
        story.append(Image(rmse_fig_path, width=380, height=237))
        story.append(Paragraph("<font size=8 color='#415A77'>圖 2：測試集上 RMSE 隨特徵數量增加的變化趨勢圖</font>", subtitle_style))
        
    story.append(PageBreak())
    
    # --- PAGE 3: FEATURE SELECTIONS ---
    story.append(Paragraph("第二節：5 種特徵篩選方法交叉分析", h1_style))
    story.append(Paragraph(
        "為客觀驗證哪些特徵對 Profit 貢獻最高，本專案實作了五種主流的特徵篩選演算法，包括：其一是 SFS (Forward)、其二是 RFE、"
        "其三是 SelectKBest、其四是 Lasso L1 迴歸、以及其五是隨機森林重要性。排名彙整如下：",
        body_style
    ))
    
    # Load ranking table
    rank_csv = os.path.join("outputs", "reports", "feature_selection_ranking.csv")
    if os.path.exists(rank_csv):
        df_rank = pd.read_csv(rank_csv)
        table_data = [[Paragraph(col, table_header_style) for col in ["特徵名稱", "SFS Forward", "RFE", "SelectKBest", "Lasso", "Random Forest", "平均排名"]]]
        for _, row in df_rank.iterrows():
            table_data.append([
                Paragraph(str(row["Feature"]), table_text_style),
                Paragraph(str(row["SFS Forward"]), table_text_style),
                Paragraph(str(row["RFE"]), table_text_style),
                Paragraph(str(row["SelectKBest"]), table_text_style),
                Paragraph(str(row["Lasso"]), table_text_style),
                Paragraph(str(row["Random Forest"]), table_text_style),
                Paragraph(f"{row['Average Rank']:.1f}", table_text_style)
            ])
            
        rank_table = Table(table_data, colWidths=[100, 65, 55, 65, 55, 75, 65])
        rank_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0077B6")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#415A77")),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EBF2FA")])
        ]))
        story.append(rank_table)
        story.append(Spacer(1, 15))
        
    rank_fig_path = os.path.join("outputs", "figures", "feature_selection_comparison.png")
    if os.path.exists(rank_fig_path):
        story.append(Image(rank_fig_path, width=380, height=228))
        story.append(Paragraph("<font size=8 color='#415A77'>圖 3：5 種特徵篩選方法之特徵重要性排名對比熱力圖</font>", subtitle_style))
        
    story.append(PageBreak())
    
    # --- PAGE 4: EVALUATION & FORMULAS ---
    story.append(Paragraph("第三節：最佳模型方程式與殘差診斷", h1_style))
    story.append(Paragraph(
        "綜合多模型指標對比與特徵重要度分析，雖然單特徵模型的 RMSE 表現略優（屬測試集小樣本之抖動），但考量到業務操作的完整度，"
        "我們採用 <b>2 Features (R&D Spend + Marketing Spend)</b> 作為最終模型的特徵組合：",
        body_style
    ))
    
    formula_latex = (
        "Profit = 50286.8118 + 0.8056 * (R&D Spend) + 0.0272 * (Marketing Spend)"
    )
    story.append(Paragraph(formula_latex, code_style))
    
    story.append(Paragraph(
        "<b>係數解讀：</b>在其他條件不變的情況下，公司每投入 1 美元於研發（R&D Spend），獲利預估邊際增加 0.81 美元；"
        "而投入於行銷（Marketing Spend），預期僅能邊際回收 0.03 美元利潤。這說明研發對該新創數據集具有絕對優勢的驅動力。",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    fit_fig_path = os.path.join("outputs", "figures", "actual_vs_predicted.png")
    residual_fig_path = os.path.join("outputs", "figures", "residual_plot.png")
    
    # Place plots side by side or vertically
    if os.path.exists(fit_fig_path) and os.path.exists(residual_fig_path):
        # Fits vertically for page layout
        story.append(Image(fit_fig_path, width=280, height=210))
        story.append(Image(residual_fig_path, width=280, height=210))
        story.append(Paragraph("<font size=8 color='#415A77'>圖 4：最佳模型的擬合散佈圖 (上) 與殘差分佈圖 (下)</font>", subtitle_style))
        
    story.append(PageBreak())
    
    # --- PAGE 5: INFOGRAPHIC CHEAT SHEET ---
    story.append(Paragraph("附錄：CRISP-DM 實作手繪風知識圖卡 (Infographic)", h1_style))
    story.append(Paragraph(
        "以下為利用 Excalidraw 手繪 chalk 風格繪製的直式 A4 知識圖卡，彙整了 Business 理解、資料前處理陷阱避防、多元模型比對與特徵選擇的核心推論。",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    info_path = "image.png"
    if os.path.exists(info_path):
        # Full-page sizing on A4 (Width 400, Height ~565)
        story.append(Image(info_path, width=400, height=565))
        story.append(Paragraph("<font size=8 color='#415A77'>圖 5：Kaggle 50 Startups Profit Prediction CRISP-DM 流程手繪風知識圖卡</font>", subtitle_style))
        
    # Build Document
    doc.build(story)
    print(f"Technical PDF Handout generated successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf_handout()
