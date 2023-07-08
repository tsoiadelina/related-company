import pandas as pd

import torch
from torch_geometric.data import HeteroData
from torch_geometric import nn
import streamlit as st


class ThreeLayers(torch.nn.Module):
    """
    Модель, которая возвращает при вызове метода forward уверенность в том, что такая связь существует

    Methods
    -------
    forward(graph) - делает проход вперед, получая эмебеддинги узлов
    predict(graph) - на эмбеддингах строит прогноз, о том сущесвует ли связь
    """

    def __init__(self, hidden_channels, num_layers=1, emb_size=256):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            nn.HeteroConv({
                ('code', 'type_0', 'code'): nn.GATConv((-1, -1), hidden_channels),
                ('code', 'type_1', 'code'): nn.GATConv((-1, -1), hidden_channels),
                ('code', 'type_2', 'code'): nn.GATConv((-1, -1), hidden_channels),
            }, aggr='mean'))
        for _ in range(num_layers - 1):
            conv = nn.HeteroConv({
                ('code', 'type_0', 'code'): nn.GATConv((-1, -1), hidden_channels),
                ('code', 'type_1', 'code'): nn.GATConv((-1, -1), hidden_channels),
                ('code', 'type_2', 'code'): nn.GATConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

        self.convs.append(
            nn.HeteroConv({
                ('code', 'type_0', 'code'): nn.GATConv((-1, -1), emb_size),
                ('code', 'type_1', 'code'): nn.GATConv((-1, -1), emb_size),
                ('code', 'type_2', 'code'): nn.GATConv((-1, -1), emb_size),
            }, aggr='mean'))

        self.lin = nn.Linear(emb_size, 1)  # Полносвязный слой для предсказания существования связи

    def forward(self, graph: HeteroData):

        x = graph.x_dict
        for conv in self.convs:
            # [print(i.shape, end='\t') for i in (x, edge_indexes, node_types, edge_types, edge_attr)]
            # [print(i.dtype, end='\t') for i in (x, edge_indexes, node_types, edge_types, edge_attr)]

            x = conv(x, graph.edge_index_dict)

        return x

    def predict(self, graph: HeteroData):
        x = self.lin(self.forward(graph)['code'])
        return torch.sigmoid(x)


@st.cache_resource
class EmbeddingOKVED:

    def __init__(self):
        self.embeddings_model = torch.load('./src/data/HeteroConv_emb.th')
        self.okved_data = pd.read_csv('./src/data/okved_2014_w_sections.csv')

        self.id_to_code = self.okved_data['native_code'].to_dict()
        self.id_to_code[0] = '0'
        self.code_to_id = {v: u for u, v in self.id_to_code.items()}

    def get_id_by_code(self, code: str) -> int:
        """
        Возвоащает код кода ОКВЭД
        :param code:
        :return:
        """
        return self.code_to_id.get(code, self.code_to_id['0'])

    def get_code_by_id(self, _id: int) -> str:
        """

        :param _id:
        :return:
        """
        return self.id_to_code.get(_id, self.id_to_code[0])

    def get_embedding(self, code: str) -> torch.Tensor:
        """
        Возвращает эмбеддинг кода ОКВЭД

        :param code:
        :return:
        """

        return self.embeddings_model[self.get_id_by_code(code)]

    def get_n_nearest(self, code: str, n: int = 5) -> list[str]:
        """
        Возвращает топ-5 близких кодов ОКВЭД к заданному
        :param code:
        :param n: Количество ближайших кодов
        :return:
        """
        emb = self.get_embedding(code)

        distances = (self.embeddings_model - emb).pow(2).sum(axis=1)
        # distances = torch.nn.functional.cosine_similarity(self.embeddings_model, emb, dim=1)

        nearest_codes = distances.topk(n + 1, largest=False).indices[1:]
        return list(map(lambda x: self.id_to_code.get(int(x)), nearest_codes))

    def get_ru_name_by_code(self, code: str) -> str:
        """
        Возвращает описание по коду

        :param code:
        :return:
        """
        return self.okved_data.loc[self.okved_data.native_code == code, 'name_okved'].values[0]

    def get_nearest_ru_names(self, code: str, n: int = 5) -> list[str]:
        """
        Возвращает топ-5 близких кодов ОКВЭД к заданному
        :param code:
        :param n: Количество ближайших кодов
        :return:
        """
        return list(map(lambda x: f'{x} : {self.get_ru_name_by_code(x)}', self.get_n_nearest(code, n)))

    def get_df_w_names(self, code: str, n: int = 5) -> pd.DataFrame:
        """

        :param code:
        :param n:
        :return:
        """
        codes = self.get_n_nearest(code, n)
        names = list(map(self.get_ru_name_by_code, codes))
        return pd.DataFrame({'Код ОКВЭД': codes, 'Описание': names})
