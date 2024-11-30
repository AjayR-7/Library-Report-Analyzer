import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask, render_template, send_file, request, redirect, url_for

# Create Flask app
app = Flask(__name__)

# Configuring the folders for uploads and static files
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the dataset (ensure the path to your Excel file is correct)
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and file.filename.endswith('.xlsx'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and process the Excel file
        df = pd.read_excel(filepath)

        # Ensure 'E_Date' column exists and convert it to datetime
        if 'E_Date' in df.columns:
            df['E_Date'] = pd.to_datetime(df['E_Date'], errors='coerce')
            df.dropna(subset=['E_Date'], inplace=True)  # Drop rows with invalid dates
        else:
            print("'E_Date' column is missing in the dataset.")

        # Feature 1: Average Time per Visit
        df['VisitDuration'] = (df['Out_Time'] - df['In_Time']).dt.total_seconds() / 60  # in minutes
        avg_time_per_visit = df['VisitDuration'].mean()

        # Feature 2: Returning vs New Visitors
        student_visit_count = df.groupby('Member ID').size()  # Grouping by 'Member ID'
        new_visitors = student_visit_count[student_visit_count == 1].count()
        returning_visitors = student_visit_count[student_visit_count > 1].count()

        # Feature 3: Seasonal Trends (Visits by season)
        df['Month'] = df['E_Date'].dt.month
        df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                               ('Spring' if x in [3, 4, 5] else
                                                ('Summer' if x in [6, 7, 8] else 'Fall')))
        seasonal_trends = df.groupby('Season').size()

        # Feature 4: Peak Visit Time Analysis (Weekly Trends)
        df['Weekday'] = df['E_Date'].dt.day_name()
        weekday_visits = df.groupby('Weekday').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        # Feature 5: Simple Prediction Model - Linear Regression
        df['DayOfYear'] = df['E_Date'].dt.dayofyear
        visit_data = df.groupby('DayOfYear').size().reset_index(name='VisitCount')

        X = visit_data[['DayOfYear']]
        y = visit_data['VisitCount']

        model = LinearRegression()
        model.fit(X, y)
        future_days = np.array([i for i in range(366, 376)]).reshape(-1, 1)  # Predict next 10 days (for a non-leap year)
        predictions = model.predict(future_days)

        # Feature 6: Clustering Students by Visit Frequency
        visit_frequency = student_visit_count.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(visit_frequency)

        cluster_df = pd.DataFrame({'Member ID': student_visit_count.index, 'Cluster': clusters})

        # Monthly Frequency Analysis for Yearly Analysis
        df['Year'] = df['E_Date'].dt.year
        df['Month'] = df['E_Date'].dt.month_name()

        # Group visits by Year and Month
        monthly_visits = df.groupby(['Year', 'Month']).size().unstack(fill_value=0)

        # Most Frequent Month Visited
        most_frequent_month = df['Month'].mode()[0]

        # Plotting

        # Bar chart for Visit Duration Categories
        bins = [0, 30, 60, 120, np.inf]
        labels = ['< 30 min', '30-60 min', '60-120 min', '> 120 min']
        df['DurationCategory'] = pd.cut(df['VisitDuration'], bins=bins, labels=labels)
        duration_counts = df['DurationCategory'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        duration_counts.plot(kind='bar', color=colors, edgecolor='black')

        plt.title('Visit Duration Categories')
        plt.xlabel('Visit Duration Range')
        plt.ylabel('Number of Visits')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'duration_categories.png'))

        # New vs Returning Visitors chart
        visitor_data = pd.DataFrame({
            'Visitor Type': ['New Visitors', 'Returning Visitors'],
            'Count': [new_visitors, returning_visitors]
        })

        plt.figure(figsize=(8, 6))
        sns.barplot(data=visitor_data, x='Visitor Type', y='Count', palette=['#66b3ff', '#ff9999'], edgecolor='black')

        plt.title('New vs Returning Visitors')
        plt.xlabel('Visitor Type')
        plt.ylabel('Number of Visitors')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'new_vs_returning_visitors.png'))

        # Seasonal Trends chart
        plt.figure(figsize=(8, 6))
        sns.barplot(x=seasonal_trends.index, y=seasonal_trends.values, palette='coolwarm')
        plt.title('Library Visits by Season')
        plt.xlabel('Season')
        plt.ylabel('Number of Visits')
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'seasonal_trends.png'))

        # Weekday Visit Analysis chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=weekday_visits.index, y=weekday_visits.values, palette='viridis')
        plt.title('Library Visits by Weekday')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Visits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'weekday_visits.png'))

        # Visit Prediction chart
        plt.figure(figsize=(10, 6))
        plt.plot(visit_data['DayOfYear'], visit_data['VisitCount'], label='Historical Data')
        plt.plot(future_days, predictions, label='Predicted Visits', color='orange')
        plt.title('Visit Prediction using Linear Regression')
        plt.xlabel('Day of Year')
        plt.ylabel('Number of Visits')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'visit_prediction.png'))

        # Clustering Students by Visit Frequency chart
        y_jitter = np.random.rand(len(visit_frequency)) * 0.5  # Adjust 0.5 for jitter amount
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=visit_frequency.flatten(), y=y_jitter, hue=cluster_df['Cluster'], palette='viridis')
        plt.title('Student Visit Frequency Clusters')
        plt.xlabel('Visit Frequency')
        plt.ylabel('Student')
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'student_visit_clusters.png'))

        # Monthly Frequency Analysis chart
        plt.figure(figsize=(12, 8))
        monthly_visits.T.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')
        plt.title('Monthly Library Visits (Yearly Analysis)')
        plt.xlabel('Month')
        plt.ylabel('Number of Visits')
        plt.xticks(rotation=45)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'monthly_visits.png'))

        # Most Frequent Month Visited chart
        month_counts = df['Month'].value_counts()
        plt.figure(figsize=(8, 6))
        month_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='tab20', legend=False)
        plt.title('Most Frequent Month Visited')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'most_frequent_month.png'))

        return redirect(url_for('show_report'))

@app.route('/report')
def show_report():
    return render_template('report.html')


# # Route to generate the PDF
# @app.route('/download_pdf')
# def download_pdf():
#     # Render the HTML report
#     rendered = render_template('report.html')

#     # Define the path for saving the PDF
#     pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'report.pdf')

#     # Use pdfkit to convert HTML to PDF
#     pdfkit.from_string(rendered, pdf_path)

#     # Send the PDF file to the user as an attachment
#     return send_file(pdf_path, as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    # Render the HTML content to be converted to PDF
    rendered_html = render_template('report.html')
    pdf_file = wkhtmltopdf.render(rendered_html)  # Convert the HTML to PDF
    
    # Return the PDF file as a downloadable attachment
    return send_file(pdf_file, as_attachment=True, download_name="Library_Report.pdf")

# Route to download generated image files
@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
