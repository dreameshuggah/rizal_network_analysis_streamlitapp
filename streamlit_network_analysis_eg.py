import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import community.community_louvain # pip install python-louvain
from pandasql import sqldf
import random

# Initial Setup
random.seed(10)

st.set_page_config(
    page_title="Network Analysis Intelligence", 
    page_icon="🕸️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- LUXURY PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* App background and text */
    .stApp {
        background-color: #0A0A0A;
        color: #E0E0E0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #2A2A2A;
    }
    
    /* Main Headers */
    h1, h2, h3, h4, h5 {
        color: #D4AF37 !important; /* Elegant Gold */
        font-weight: 300 !important;
        letter-spacing: 0.5px;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #2A2A2A;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Styling Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #A0A0A0;
        font-weight: 500;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 2px solid #D4AF37 !important;
        color: #D4AF37 !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Multiselect and inputs */
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #1A1A1A;
        border: 1px solid #333;
        border-radius: 6px;
    }

    /* Success/Info text */
    .st-emotion-cache-1kyxreq {
        color: #D4AF37;
    }
    
    /* Hide default header */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #D4AF37;'>Data Configuration</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: #888;'>Upload your network dataset to begin</p>", unsafe_allow_html=True)
    st.divider()
    
    sample_data = st.checkbox('Use sample data.')
    
    if sample_data:
        uploaded_file = 'players.csv'
    else:
        uploaded_file = st.file_uploader("Choose a CSV file")
        
    if uploaded_file is None: 
        st.info('Please Upload a CSV file to proceed.')
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        uniq = sorted(list(set([*df['from'],*df['to']])))
        relationships = sorted(list(df['label'].unique()))


tab1, tab2 = st.tabs(["🌐 Community Detection", "👤 Individual Networks"])

with tab1:
    st.markdown("<h1>Network Analysis: <span style='color: #E0E0E0;'>Community Detection</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em; color: #A0A0A0;'>Analyze global network structures and discover inherent communities within your data.</p>", unsafe_allow_html=True)
    st.divider()

    if uploaded_file is None: 
        st.warning('Awaiting dataset upload...')
    else:
        # Prevent pandasql from capturing globals that might conflict
        agg_label_df = sqldf("""
                      SELECT 
                      label
                      ,COUNT(label) AS counts
                      FROM df
                      GROUP BY label
                      ORDER BY counts DESC
                      """, locals())
        
        agg_from_df = sqldf("""
                      SELECT 
                      `from`
                      ,label
                      ,COUNT(`to`) AS counts
                      FROM df
                      GROUP BY `from`
                      ,label
                      ORDER BY counts DESC
                      """, locals())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('#### 📄 Input Data Overview')
            st.dataframe(df, use_container_width=True)
            
        with col2:
            st.markdown('#### 📊 Aggregated Label Counts')
            st.dataframe(agg_label_df, use_container_width=True)
    
        st.divider()
        st.markdown('#### 🎯 Relationship Filtering')
        
        default_rel = ['rival'] if 'rival' in relationships else (relationships[:1] if len(relationships) > 0 else [])
        relation = st.multiselect('Select Labels to Visualize', relationships, default=default_rel)
        
        if len(relation) == 0:
            st.info('Choose at least 1 label to generate the network map.')
        else:
            df_select = df.loc[df['label'].isin(relation)].reset_index(drop=True)
            
            G = nx.from_pandas_edgelist(df_select, 
                                    source="from", 
                                    target="to", 
                                    edge_attr=True, 
                                    create_using=nx.Graph())
        
            communities = community.community_louvain.best_partition(G, random_state=123) 
            communities_df = pd.DataFrame.from_dict(communities, orient='index', columns=['community']).reset_index()
            communities_df = communities_df.rename(columns={'index':'from'})
            
            df_select2 = sqldf("""
                                SELECT 
                                a.`from`
                                ,a.`to`
                                ,a.label
                                ,a.weight
                                ,b.community
                                FROM df_select a
                                LEFT JOIN communities_df b
                                ON a.`from` = b.`from`
                                """, locals())
      
            agg_community_df = sqldf("""
                            SELECT 
                            `from`
                            ,label
                            ,COUNT(label) AS counts
                            ,community
                            ,GROUP_CONCAT(`to`) AS connected_parties
                            FROM df_select2
                            GROUP BY `from`,label,community
                            ORDER BY counts DESC
                            """, locals())
            
            st.markdown('#### 🏢 Community Detection Results')
            st.dataframe(agg_community_df, use_container_width=True)
            
            uniq_community = sorted(list(agg_community_df['community'].unique()))
            
            if len(uniq_community) > 0:
                st.divider()
                st.markdown('#### 🌌 Visualization Configuration')
                select_community = st.multiselect('Select Communities to Visualize', uniq_community)
                
                if len(select_community) == 0:
                    st.info('Choose at least 1 community group to launch visualization.')
                else:
                    dfg = df_select2.loc[df_select2['community'].isin(select_community)].copy()
                    dfg['title'] = dfg['community']
                    
                    dfg_tbl = dfg[['from', 'to', 'label', 'community']].copy()
                    dfg = dfg.rename(columns={'community': 'group'})
                    
                    G_vis = nx.from_pandas_edgelist(dfg, 
                                    source="from", 
                                    target="to", 
                                    edge_attr=True, 
                                    create_using=nx.Graph())
      
                    net = Network(notebook=True,
                                  cdn_resources='remote',
                                  width='100%',
                                  height="800px",
                                  select_menu=True,
                                  filter_menu=True,
                                  bgcolor='#0A0A0A',
                                  font_color='#E0E0E0')
      
                    node_degree = dict(G_vis.degree)
                    nx.set_node_attributes(G_vis, node_degree, 'size')
                    
                    nx.set_node_attributes(G_vis, communities, 'group')
      
                    net.from_nx(G_vis)
                    net.repulsion(node_distance=420,
                                central_gravity=0.33,
                                spring_length=110,
                                spring_strength=0.10,
                                damping=0.95)

                    outFile1 = 'community.html'
                    net.save_graph(outFile1)
                    
                    st.markdown("""
                        <div style="background-color: #121212; padding: 15px; border-radius: 8px; border: 1px solid #2A2A2A; margin-bottom: 20px;">
                            <h5 style="margin-top: 0; color: #D4AF37;">💡 Navigation Guide</h5>
                            <ul style="color: #A0A0A0; padding-left: 20px; margin-bottom: 0;">
                                <li>Scroll Mouse up/down to zoom in/out.</li>
                                <li>Nodes are colored by Community Detection.</li>
                                <li>Edge width reflects connection Weight.</li>
                                <li>Node size denotes the number of connected links.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with open(outFile1, 'r', encoding='utf-8') as HtmlFile:
                        components.html(HtmlFile.read(), height=850)
                    
                    dfg_tbl = dfg_tbl.rename(columns={'label': 'relationship'})
                    st.divider()
                    st.markdown('#### 📝 Selected Communities Data')  
                    st.dataframe(dfg_tbl, use_container_width=True)

with tab2:
    st.markdown("<h1>Network Analysis: <span style='color: #E0E0E0;'>1st Degree Networks</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em; color: #A0A0A0;'>Isolate and examine direct connections for specific individuals.</p>", unsafe_allow_html=True)
    st.divider()
    
    if uploaded_file is None: 
        st.warning('Awaiting dataset upload...')
    else:
        st.markdown('#### 🎯 Target Identification')
        individual_ls = st.multiselect('Select Individuals to visualize', uniq)    
    
        if len(individual_ls) == 0:
            st.info('Choose at least 1 individual to build standard network.')
        else:
            individual_df = df[(df['from'].isin(individual_ls)) | (df['to'].isin(individual_ls))]
      
            G_ind = nx.from_pandas_edgelist(individual_df, 
                            source="from", 
                            target="to", 
                            edge_attr=True, 
                            create_using=nx.Graph())
    
            net_ind = Network(notebook=True,
                         cdn_resources='remote',
                         width='100%',
                         height="800px",
                         select_menu=True,
                         filter_menu=True,
                         bgcolor='#0A0A0A', 
                         font_color='#E0E0E0')
            
            node_degree_ind = dict(G_ind.degree)
            nx.set_node_attributes(G_ind, node_degree_ind, 'size')
                
            communities_ind = community.community_louvain.best_partition(G_ind, random_state=123)
            nx.set_node_attributes(G_ind, communities_ind, 'group')
                
            net_ind.from_nx(G_ind)
            net_ind.repulsion(node_distance=420,
                           central_gravity=0.33,
                           spring_length=110,
                           spring_strength=0.10,
                           damping=0.95)
                
            outFile2 = 'individual.html'
            net_ind.save_graph(outFile2)
            
            with open(outFile2, 'r', encoding='utf-8') as HtmlFile2:
                components.html(HtmlFile2.read(), height=850)
                
            st.divider()
            st.markdown('#### 📝 Isolated Network Data')    
            st.dataframe(individual_df, use_container_width=True)






