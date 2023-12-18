from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'
## RETRIEVER
api_key = os.environ["WEAVIATE_API_KEY"]
url = os.environ["WEAVIATE_ENDPOINT"]

client = WeaviateClient(api_key, url)
available_classes = client.show_classes()
## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
## LLM 
llm = GPT_Turbo(model='gpt-3.5-turbo-0613', api_key=os.environ["OPENAI_API_KEY"])
## ENCODING
encoding = encoding_for_model('gpt-3.5-turbo-0613')
## INDEX NAME
class_name = 'impact_theory_minl_256'
##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest_input = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

        alpha_input = st.slider('Alpha for Hybrid Search', 0.00, 1.00, step=0.45)
        retrieval_limit = st.slider('Limit for retrieval results', 1, 100, 10)
        reranker_topk = st.slider('Top K for Reranker', 1, 50, 3)
        temperature_input = st.slider('Temperature for LLM', 0.0, 2.0, 1.0)

        class_name = st.selectbox('Class Name:', options=available_classes, index=None, placeholder='Select Class Name')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest_input).todict() if guest_input else None

            # make hybrid call to weaviate
            hybrid_response = client.hybrid_search(query,
                                                   class_name=class_name,
                                                   alpha=alpha_input,
                                                   display_properties=client.display_properties,
                                                   where_filter=guest_filter,
                                                   limit=retrieval_limit)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response,
                                              query,
                                              apply_sigmoid=True,
                                              top_k=reranker_topk)
            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response, question_answering_prompt_series,
                                     query=query,
                                     tokenizer=encoding,  # variable from ENCODING,
                                     token_threshold= 4000,
                                     verbose=True)
            ############## 
            #  END CODE  #
            ##############

            # # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)
            
            # # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()
                for resp in llm.get_chat_completion(prompt=prompt,
                                                    system_message='Answer this question using the material provided in the prompt.',
                                                    temperature=temperature_input, max_tokens=500, stream=True):
                    try:
                        with response_box:
                            content = resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                st.write(f'{result}')
                    except Exception as e:
                        print(e)
                        continue               ##############
            # ##############
            # # START CODE #
            # ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length']
                time_string = convert_seconds(show_length)
            # ##############
            # #  END CODE  #
            # ##############
            with col1:
             st.write(search_result(i=i, 
                                    url=episode_url,
                                    guest=hit['guest'],
                                    title=title,
                                    content=hit['content'], 
                                    length=time_string),
                                    unsafe_allow_html=True)
             st.write('\n\n')
            with col2:
                #st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                 #        unsafe_allow_html=True)
                st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()