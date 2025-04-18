def generate_report(df, visualizations, batch_id):
    """Generate an HTML report from analysis results."""
    import pandas as pd
    
    # Start HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speech Analysis Report - Batch {batch_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                      background-color: #f8f9fa; border-radius: 5px; width: 220px; }}
            .metric h3 {{ margin-top: 0; color: #3498db; }}
            .visualization {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Speech Analysis Report</h1>
            <p>Batch ID: {batch_id}</p>
            <p>Samples Analyzed: {len(df)}</p>
            
            <h2>Key Metrics</h2>
            <div>
                <div class="metric">
                    <h3>Average Risk Score</h3>
                    <p>{df['risk_score'].mean():.1f}/100</p>
                </div>
                <div class="metric">
                    <h3>High Risk Samples</h3>
                    <p>{len(df[df['risk_score'] > 70])}/{len(df)}</p>
                </div>
                <div class="metric">
                    <h3>Anomalies Detected</h3>
                    <p>{df['is_anomaly'].sum()}/{len(df)}</p>
                </div>
                <div class="metric">
                    <h3>Avg Speaking Rate</h3>
                    <p>{df['wpm'].mean():.1f} WPM</p>
                </div>
            </div>
    """
    
    # Add visualizations
    if visualizations:
        html += """
            <h2>Visualizations</h2>
            <div class="visualizations">
        """
        
        # Add correlation heatmap
        if 'correlation_heatmap' in visualizations:
            html += f"""
                <div class="visualization">
                    <h3>Feature Correlation Heatmap</h3>
                    <img src="data:image/png;base64,{visualizations['correlation_heatmap']}" 
                         alt="Correlation Heatmap" width="100%">
                </div>
            """
            
        # Add risk score distribution
        if 'risk_score_distribution' in visualizations:
            html += f"""
                <div class="visualization">
                    <h3>Risk Score Distribution</h3>
                    <img src="data:image/png;base64,{visualizations['risk_score_distribution']}" 
                         alt="Risk Score Distribution" width="100%">
                </div>
            """
            
        html += "</div>"
    
    # Add sample table
    html += """
        <h2>Sample Results</h2>
        <table>
            <tr>
                <th>Sample ID</th>
                <th>Risk Score</th>
                <th>Speaking Rate (WPM)</th>
                <th>Pause Rate</th>
                <th>Hesitation Rate</th>
                <th>Word Finding Difficulty</th>
                <th>Anomaly</th>
            </tr>
    """
    
    # Add rows for each sample
    for _, row in df.iterrows():
        html += f"""
            <tr>
                <td>{row.get('sample_id', 'Unknown')}</td>
                <td>{row.get('risk_score', 0):.1f}</td>
                <td>{row.get('wpm', 0):.1f}</td>
                <td>{row.get('pause_rate', 0):.3f}</td>
                <td>{row.get('hesitation_rate', 0):.3f}</td>
                <td>{row.get('word_finding_difficulty', 0):.3f}</td>
                <td>{"Yes" if row.get('is_anomaly', False) else "No"}</td>
            </tr>
        """
    
    # Close table and HTML document
    html += """
        </table>
        </div>
    </body>
    </html>
    """
    
    return html
