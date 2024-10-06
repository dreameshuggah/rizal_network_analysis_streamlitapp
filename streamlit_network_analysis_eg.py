#!/usr/bin/env python
# coding: utf-8

# In[2]:

# cd /Users/dreameshuggah/Documents/Rizal_Analytics/Network_Analysis/Rizal_Network_Analysis_StreamLitApp
# streamlit run streamlit_network_analysis_eg.py 
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
# In[4]:


st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)



### READ: DATA
df = pd.read_csv('invite2.csv',usecols=['from','to'])
#df['weight']=1
labels = ['friend','rival']
df['label'] = random.choices(labels,weights=None,k=len(df))
weight = [1,2,3,4,5,6,7,8]
df['weight'] = random.choices(weight,weights=None,k=len(df))
df['title'] =  df['label']






# Set header title
st.title('Community Detection of Players')






col1, col2 = st.columns(2)





col1.text('\n\n\n')
col1.markdown('Raw Dataset')
col1.dataframe(df[['from','to','label','weight']].rename(columns={'label':'relationship','weight':'n_games_played'}))


agg_label_df = sqldf("""
                SELECT 
                label
                ,COUNT(label) AS counts
                ,AVG(weight) AS average_games_played
                FROM df
                GROUP BY label
                ORDER BY counts DESC

                """,locals())
#agg_label_df

col2.text('\n\n\n')
col2.markdown('Relationship Statistics:')
col2.dataframe(agg_label_df.rename(columns={'label':'relationship'}))#,use_container_width=True)

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
#agg_from_df

uniq = sorted(list(set([*df['from'],*df['to']])))

relationships= sorted(list(df['label'].unique()))









# Define list of selection options and sort alphabetically
# Implement multiselect dropdown menu for option selection (returns a list)
st.text('\n\n\n')
st.text('\n\n\n')
relation = st.multiselect('Select relationships to visualize', relationships,['friend','rival'])


# In[ ]:


# Set info message on initial site load
if len(relation) == 0:
    st.text('\n\n\n')
    st.text('\n\n\n')
    st.text('Choose at least 1 relationship to start')

# Create network graph when user selects >= 1 item
else:
    df_select = df.loc[df['label'].isin(relation) ]
    df_select = df_select.reset_index(drop=True)



    G = nx.from_pandas_edgelist(df_select, 
                            source = "from", 
                            target = "to", 
                            edge_attr = True, 
                            create_using = nx.Graph()
                               )

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
                        ,b.community AS title
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
                    ,GROUP_CONCAT(`to`) AS connected_players
                    FROM df_select2
                    GROUP BY `from`,label,community
                    ORDER BY counts DESC
                    """,locals())


    #st.text('\n\n\n')
    st.markdown('\nTable of Communities based on relationships')
    st.dataframe(agg_community_df.rename(columns={'label':'relationship'}),use_container_width=True)
    
    
    
    uniq_community = sorted(list(agg_community_df['community'].unique()))
    
    
    
    if len(uniq_community) > 0:
        st.text('\n\n\n')
        st.text('\n\n\n')
        select_community = st.multiselect('Select community to visualize'
                                          , uniq_community
                                          , agg_community_df['community'][:2].values
                                         )
        
        if len(select_community)==0:
            st.text('Choose at least 1 community to start')
        
        else:
            dfg = df_select2.loc[df_select2['community'].isin(select_community)]
            #dfg['title'] = dfg['community'].values
            

            dfg_tbl = dfg[['from','to','label','community']].copy()
            dfg = dfg.rename(columns={'community':'group'})
            
            G = nx.from_pandas_edgelist(dfg, 
                            source = "from", 
                            target = "to", 
                            edge_attr = True, 
                            create_using = nx.DiGraph()  #nx.Graph()
                                       )

            # Initiate PyVis network object
            net = Network(notebook = True
                          ,cdn_resources='remote'
                          ,width= '100%'#width="1000px"
                          ,height="1000px"
                          ,select_menu=True
                          #,filter_menu=True
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
            net.repulsion(
                        node_distance=420,
                        central_gravity=0.33,
                        spring_length=110,
                        spring_strength=0.10,
                        damping=0.95
                       )

            net.save_graph('test_streamlit_pyvis.html')
            
            mytext = """
            - Scroll Mouse up/down to zoom in/out
            - Color by community detection
            - Size of edges by n_games_played together
            
            """
            
            st.markdown(mytext)
            HtmlFile = open('test_streamlit_pyvis.html', 'r', encoding='utf-8')



            # Load HTML file in HTML component for display on Streamlit page
            components.html(HtmlFile.read(), height=1000)
            
            dfg_tbl = dfg_tbl.rename(columns={'label':'relationship'})

            st.text('\n\n\n')
            st.text('\n\n\n')
            st.markdown('Table of Selected Communities:')
            st.dataframe(dfg_tbl.sort_values(by=['community','relationship']),use_container_width=True)
    
    
    


   




