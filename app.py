# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#####################################
########## INITIAL SETUP ############
#####################################

# loading styles - costum css
def load_css(file_name):
    with open(file_name) as f:
        st.html(f"<style>{f.read()}</style>")

load_css("style.css")


# page configuration
st.set_page_config(
    page_title = "Application Servier-Symphogen",
    page_icon = ":wave:",
    layout = "wide",
)



#####################################
########## THE APPLICATION ##########
#####################################

# Header and subheader
st.header("Application for Servier-Symphogen")
st.write("Dear Servier-Symphogen. As mentioned in my application, I have just started developing my skills using streamlit for app generation.\n I have experience in doing the same using the Shiny library in R.")


###### Caching image ######

#defining function for caching image ensuring faster loading
def load_image(img_path):
    return Image.open(img_path)




#### SIDEBAR NAVIGATION ####

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["About Me", "Demo: Gene Expression Explorer"])


# ==================== ABOUT ME PAGE ====================

# Could add
# ["Relevant Courses", "Resumé highlights", "Why I want to work at Servier-Symphogen?"])


if page == "About Me":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### About Me")
        st.write("""
        My name is Kristine and I'm currently studying on my second year of Bioinformatics at DTU.
        I have a strong passion for computational biology, particularly in the areas of immunology and protein design.
        My courses this semester include High Performance Computing and a special course, where I am designing high specificity de novo proteins targeting a cancer antigen. 
        """)

        st.markdown("### Highlighted Courses")
        st.write("""
        - High-Performance Computing (current)
        - Machine Learning
        - Special course: De novo Protein Design (current)
        - Immunology
        - Molecular Biology
        """)
        
        st.markdown("### Highlighted Computational Skills")
        st.write("""
        - Python
        - R
        - App development with Shiny + currently learning Streamlit
        - Unix/Linuz command line
        - Version control with Git/GitHub
        """)

    with col2:
        st.markdown("### Connect With Me")
        st.info("""
        Email: kristine@toftjohansen.com
        LinkedIn: https://www.linkedin.com/in/kristine-toft-johansen/
        Phone: +45 29875078
        """)
        st.markdown("---")
        
        st.markdown("### Why Servier-Symphogen?")
        # st.info("""
        # <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #0c5460;">
        # This job post fits amazingly with my interests and skills. Seems like the <span style="color: #0c5460; font-weight: bold; font-size: 1.1em;">perfect match</span>!
        # """)
        st.markdown(
        """
        <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 4px solid #0c5460;">
        This job post fits amazingly with my interests and skills. Seems like the <span style="color: #0c5460; font-weight: bold; font-size: 1.1em;">perfect match</span>!
        """, 
        unsafe_allow_html=True)





# ==================== DEMO PAGE ====================
else:
    st.markdown('<p class="main-header">Gene Expression Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive tool for cancer genomics data analysis</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate synthetic cancer gene expression data
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        
        # Cancer types
        cancer_types = ['Breast', 'Lung', 'Colon', 'Melanoma', 'Prostate']
        n_samples_per_type = 20
        
        # Gene names (oncology/immunology relevant)
        genes = ['PD-L1', 'CTLA4', 'CD274', 'HER2', 'EGFR', 'KRAS', 
                'TP53', 'BRCA1', 'BRAF', 'CD8A', 'IFNG', 'TNF']
        
        data = []
        for cancer in cancer_types:
            for i in range(n_samples_per_type):
                sample = {'Cancer_Type': cancer, 'Sample_ID': f'{cancer}_{i+1}'}
                # Generate expression values with some cancer-specific patterns
                for gene in genes:
                    base_expr = np.random.normal(5, 1.5)
                    if gene == 'HER2' and cancer == 'Breast':
                        base_expr += np.random.normal(3, 0.5)
                    elif gene == 'EGFR' and cancer == 'Lung':
                        base_expr += np.random.normal(2, 0.5)
                    sample[gene] = max(0, base_expr)
                data.append(sample)
        
        return pd.DataFrame(data), genes
    
    df, gene_list = generate_sample_data()
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")
    selected_cancers = st.sidebar.multiselect(
        "Select Cancer Types",
        options=df['Cancer_Type'].unique(),
        default=df['Cancer_Type'].unique()
    )
    
    selected_genes = st.sidebar.multiselect(
        "Select Genes to Analyze",
        options=gene_list,
        default=gene_list[:4]
    )
    
    # Filter data
    filtered_df = df[df['Cancer_Type'].isin(selected_cancers)]
    
    if len(selected_genes) > 0 and len(selected_cancers) > 0:
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Expression Levels", "🔥 Heatmap", "🤖 ML Clustering", "📈 Statistics"])
        
        # TAB 1: Expression levels
        with tab1:
            st.markdown("### Gene Expression Distribution")
            
            plot_data = filtered_df.melt(
                id_vars=['Cancer_Type', 'Sample_ID'],
                value_vars=selected_genes,
                var_name='Gene',
                value_name='Expression'
            )
            
            fig = px.box(plot_data, x='Gene', y='Expression', color='Cancer_Type',
                        title='Gene Expression Levels by Cancer Type',
                        labels={'Expression': 'Expression Level (log2 TPM)'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sample data
            st.markdown("### 📋 Sample Data")
            st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # TAB 2: Heatmap
        with tab2:
            st.markdown("### Expression Heatmap")
            
            # Prepare data for heatmap
            heatmap_data = filtered_df.groupby('Cancer_Type')[selected_genes].mean()
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdBu_r',
                text=heatmap_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Average Gene Expression per Cancer Type',
                xaxis_title='Gene',
                yaxis_title='Cancer Type',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: ML Clustering
        with tab3:
            st.markdown("### Machine Learning: PCA & Clustering")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 5, 3)
                st.info("""
                This analysis uses:
                - **PCA** for dimensionality reduction
                - **K-Means** for clustering
                """)
            
            with col2:
                # Prepare data for ML
                X = filtered_df[selected_genes].values
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Plot
                plot_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cancer Type': filtered_df['Cancer_Type'].values,
                    'Cluster': [f'Cluster {i+1}' for i in clusters]
                })
                
                fig = px.scatter(plot_df, x='PC1', y='PC2', 
                               color='Cancer Type', 
                               symbol='Cluster',
                               title=f'PCA Visualization with {n_clusters} K-Means Clusters',
                               labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                      'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'})
                fig.update_traces(marker=dict(size=10))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Total variance explained:** {pca.explained_variance_ratio_.sum():.1%}")
        
        # TAB 4: Statistics
        with tab4:
            st.markdown("### 📊 Summary Statistics")
            
            stats_df = filtered_df.groupby('Cancer_Type')[selected_genes].agg(['mean', 'std'])
            stats_df.columns = [f'{gene}_{stat}' for gene, stat in stats_df.columns]
            
            st.dataframe(stats_df.round(2), use_container_width=True)
            
            st.markdown("### 🔬 Interpretation Notes")
            st.write("""
            - **High expression** (>7): Potential therapeutic target
            - **Variable expression** (high std): Heterogeneous population
            - **Clustering patterns**: May indicate molecular subtypes
            """)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name="gene_expression_data.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("⚠️ Please select at least one cancer type and one gene from the sidebar.")
    
    st.markdown("---")
    st.markdown("""
    **About this demo:** This tool demonstrates interactive data exploration capabilities 
    for cancer genomics research. It showcases data visualization, filtering, ML analysis, 
    and export functionality - all essential features for computational biology applications.
    """)
