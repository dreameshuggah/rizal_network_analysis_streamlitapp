#!/usr/bin/env python
# coding: utf-8

# In[2]:

# cd C:\Latize\Rizal_Analytics\Network_Analysis
# streamlit run streamlit_network_analysis_V2.py 
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain #pip install python-louvain

# Read dataset (CSV)
#df = pd.read_csv('processed_drug_interactions.csv')
#df.head()

import pandas as pd
import numpy as np

from pandasql import sqldf

import random
random.seed(10)




st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)


with st.sidebar:
    sample_data = st.checkbox('Use sample data.')
    
    if sample_data:
        uploaded_file = 'players.csv'
        
    else:
        uploaded_file = st.file_uploader("Choose a CSV file")#, accept_multiple_files=True)
    
    if uploaded_file is None: 
        st.write('Please Upload CSV file...')
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        uniq = sorted(list(set([*df['from'],*df['to']])))
        relationships= sorted(list(df['label'].unique()))


tab1, tab2 = st.tabs(["Community", "Individual"])

with tab1:
    st.title('Network Analysis: Community Detection')

    if uploaded_file is None: 
        st.write('Please Upload CSV file...')
    
    else:
        agg_label_df = sqldf("""
                      SELECT 
                      label
                      ,COUNT(label) AS counts
                      FROM df
                      GROUP BY label
                      ORDER BY counts DESC
    
                      """,locals())
      #agg_label_df
      
        
    
        agg_from_df = sqldf("""
                      SELECT 
                      `from`
                      ,label
                      ,COUNT(`to`) AS counts
                      FROM df
                      GROUP BY `from`
                      ,label
                      ORDER BY counts DESC
        
                      """,locals())
        
        st.markdown('###')
        
        col1, col2 = st.columns(2)
        
        col1.markdown('Input Data:')
        col1.dataframe(df)#,use_container_width=True)
        
        #st.markdown('###')
        col2.markdown('Aggregated Label Counts:')
        col2.dataframe(agg_label_df)#,use_container_width=True)
    
          # Define list of selection options and sort alphabetically
          # Implement multiselect dropdown menu for option selection (returns a list)
        st.markdown('###')
        st.markdown('###')
        relation = st.multiselect('Select Labels to Visualize', relationships,'rival')
        
        
          # In[ ]:
        
        
          # Set info message on initial site load
        if len(relation) == 0:
              st.text('Choose at Least 1 Label to Start')
        
          # Create network graph when user selects >= 1 item
        else:
              df_select = df.loc[df['label'].isin(relation) ]
              df_select = df_select.reset_index(drop=True)
        
        
        
              G = nx.from_pandas_edgelist(df_select, 
                                      source = "from", 
                                      target = "to", 
                                      edge_attr = True, 
                                      create_using = nx.Graph())
          
              communities =  community.community_louvain.best_partition(G,random_state=123) 
              communities_df = pd.DataFrame.from_dict(communities, orient='index', columns=['community']).reset_index()
              communities_df = communities_df.rename(columns={'index':'from'})
              
              df_select2= sqldf("""
                                  SELECT 
                                  a.`from`
                                  ,a.`to`
                                  ,a.label
                                  ,a.weight
                                  ,b.community
                                  FROM df_select a
                                  LEFT JOIN communities_df b
                                  ON a.`from` = b.`from`
                                  """,locals())
        
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
                              """,locals())
              
              st.dataframe(agg_community_df,use_container_width=True)
              
              
              
              uniq_community = sorted(list(agg_community_df['community'].unique()))
              
              
              
              if len(uniq_community) > 0:
                  st.markdown('###')
                  st.markdown('###')
                  select_community = st.multiselect('Select Communities to Visualize', uniq_community)
                  
                  if len(select_community)==0:
                      st.text('Choose at least 1 community to start')
                  
                  else:
                      dfg = df_select2.loc[df_select2['community'].isin(select_community)]
                      dfg['title'] = dfg['community']
                      
        
                      dfg_tbl = dfg[['from','to','label','community']].copy()
                      dfg = dfg.rename(columns={'community':'group'})
                      
                      G = nx.from_pandas_edgelist(dfg, 
                                      source = "from", 
                                      target = "to", 
                                      edge_attr = True, 
                                      create_using = nx.Graph())
        
                      # Initiate PyVis network object
                      net = Network(notebook = True
                                    ,cdn_resources='remote'
                                    ,width= '100%'#width="1000px"
                                    ,height="1000px"
                                    ,select_menu=True
                                    ,filter_menu=True
                                    , bgcolor='#222222', font_color='white'
                                   )
        
                      node_degree = dict(G.degree)
        
                      #Setting up node size attribute
                      nx.set_node_attributes(G, node_degree, 'size')
        
                      #import community as community_louvain
                      #communities =  community.community_louvain.best_partition(G) #community_louvain.best_partition(G)
                      nx.set_node_attributes(G, communities, 'group')
        
                      # Take Networkx graph and translate it to a PyVis graph format
                      net.from_nx(G)
        
                      # Generate network with specific layout settings
                      net.repulsion(node_distance=420,
                                  central_gravity=0.33,
                                  spring_length=110,
                                  spring_strength=0.10,
                                  damping=0.95)

                      outFile1 = 'community.html'
                      net.save_graph(outFile1)
                      mytext = """
                                - Scroll Mouse up/down to zoom in/out.
                                - Color by Community Detection.
                                - Edge Size by Weight (numerical column).
                                - Node Size by number of connected links.

                                """

                      st.markdown(mytext)
                      HtmlFile = open(outFile1, 'r', encoding='utf-8')
        
        
                      st.markdown('###')
                      # Load HTML file in HTML component for display on Streamlit page
                      components.html(HtmlFile.read(), height=1000)
                      
                      dfg_tbl = dfg_tbl.rename(columns={'label':'relationship'})
                      st.markdown('###')
                      st.markdown('Selected Communities:')  
                      st.dataframe(dfg_tbl,use_container_width=True)
    
    
    

with tab2:
    
    # Set header title
    st.title('Network Analysis: Individuals 1st Degree Networks')
    
    if uploaded_file is None: 
        st.write('Please Upload CSV file...')
    
    else:
        individual_ls = st.multiselect('Select Individuals to visualize', uniq)    
    
    
        if len(individual_ls) == 0:
            st.text('Choose at least 1 individual to start')

          # Create network graph when user selects >= 1 item
        else:
            individual_df = df[ (df['from'].isin(individual_ls)) | (df['to'].isin(individual_ls))]
      
            G = nx.from_pandas_edgelist(individual_df, 
                            source = "from", 
                            target = "to", 
                            edge_attr = True, 
                            create_using = nx.Graph()
                           )
    
            # Initiate PyVis network object
            net = Network(notebook = True
                         ,cdn_resources='remote'
                         ,width= '100%'#width="1000px"
                         ,height="1000px"
                         ,select_menu=True
                         ,filter_menu=True
                         , bgcolor='#222222', font_color='white'
                         )
            
            node_degree = dict(G.degree)
                
                #Setting up node size attribute
            nx.set_node_attributes(G, node_degree, 'size')
                
            import community as community_louvain
            communities =  community.community_louvain.best_partition(G) #community_louvain.best_partition(G)
            nx.set_node_attributes(G, communities, 'group')
                
                # Take Networkx graph and translate it to a PyVis graph format
            net.from_nx(G)
                
                # Generate network with specific layout settings
            net.repulsion(node_distance=420,
                             central_gravity=0.33,
                              spring_length=110,
                           spring_strength=0.10,
                           damping=0.95)
                
            outFile2 = 'individual.html'
            net.save_graph(outFile2)
            HtmlFile2 = open(outFile2, 'r', encoding='utf-8')
            components.html(HtmlFile2.read(), height=1000)
            st.dataframe(individual_df,use_container_width=True)

   






