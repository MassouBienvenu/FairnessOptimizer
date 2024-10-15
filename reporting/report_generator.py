import io
import os
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate_report(self, initial_fairness, final_fairness, config, original_data, adjusted_data, optimized_fairness_score):
        report_path = os.path.join(os.getcwd(), 'fairness_report.pdf')
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []

        # Add a cover page
        cover_style = ParagraphStyle(name='Cover', fontSize=24, alignment=TA_CENTER)
        story.append(Paragraph("Fairness Optimization Report", cover_style))
        story.append(Spacer(1, 100))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(PageBreak())

        # Add content
        story.append(Paragraph("Fairness Scores", self.styles['Heading1']))
        story.append(Spacer(1, 12))
        
        data = [
            ["Metric", "Score"],
            ["Initial Fairness", f"{initial_fairness:.4f}"],
            ["Optimized Fairness", f"{optimized_fairness_score:.4f}"],
            ["Final Fairness", f"{final_fairness:.4f}"]
        ]
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        story.append(Paragraph("Configuration", self.styles['Heading1']))
        story.append(Spacer(1, 12))
        for key, value in config.items():
            story.append(Paragraph(f"{key}: {value}", self.styles['Normal']))
        story.append(Spacer(1, 24))

        story.append(Paragraph("Distribution Comparison", self.styles['Heading1']))
        story.append(Spacer(1, 12))
        img_buffer = self._generate_distribution_plot(original_data, adjusted_data, config['sensitive_attributes'])
        img = Image(img_buffer)
        img.drawHeight = 450
        img.drawWidth = 600
        story.append(img)

        doc.build(story)
        return report_path

    def _generate_distribution_plot(self, original_data, adjusted_data, sensitive_attributes):
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35
        
        all_categories = set()
        for attr in sensitive_attributes:
            all_categories.update(original_data[attr].unique())
            all_categories.update(adjusted_data[attr].unique())
        
        x = np.arange(len(all_categories))
        
        original_counts = {cat: 0 for cat in all_categories}
        adjusted_counts = {cat: 0 for cat in all_categories}
        
        for attr in sensitive_attributes:
            for cat in all_categories:
                original_counts[cat] += (original_data[attr] == cat).sum()
                adjusted_counts[cat] += (adjusted_data[attr] == cat).sum()
        
        original_values = [original_counts[cat] for cat in all_categories]
        adjusted_values = [adjusted_counts[cat] for cat in all_categories]
        
        ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
        ax.bar(x + width/2, adjusted_values, width, label='Adjusted', alpha=0.8)
        
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Sensitive Attributes')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return buf