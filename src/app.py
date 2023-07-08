import streamlit

from hetero_conv_model import *
from db_connect import *
from datetime import datetime


def page_start():
    st.set_page_config(
        page_title="Код ОКВЭД",
        page_icon="🧊",
        initial_sidebar_state="auto"
    )

    st.header('Предложим код ОКВЭД вашему бизнесу!')

    if 'codes_okved' not in st.session_state:
        okved_df = pd.read_csv('./src/data/okved_2014_w_sections.csv')
        st.session_state['code_okved'] = okved_df.native_code + ' : ' + okved_df.name_okved

    # if 'okved_model' not in st.session_state:
    st.session_state['okved_model'] = EmbeddingOKVED()

    if 'db_connect' not in st.session_state:
        st.session_state['db'] = DBConnection()


def okved_choiser():
    with st.sidebar:
        num_okveds = st.number_input('Количество кодов для вывода', 1, 20, 5, 1)

    okved_option = st.selectbox('Введите один код ОКВЭД', st.session_state['code_okved'])
    similar_okveds_df = st.session_state['okved_model'].get_df_w_names(okved_option.split(' : ')[0], num_okveds)
    st.write(similar_okveds_df)

    st.session_state['db'].insert(
        (datetime.now(), okved_option.split(' : ')[0], num_okveds, similar_okveds_df.iloc[0, 0]))


page_start()
okved_choiser()
