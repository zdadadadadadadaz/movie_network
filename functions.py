#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 02:42:25 2017

@author: hnakamura
"""

# Scraiptest.pyの改訂版

# import MeCab
import urllib.request
import CaboCha
import re
import networkx as nx
from bs4 import BeautifulSoup
import sys
from typing import Any
from datapolish import *
import csv
import os
import numpy as np

def get_soup(url: str):
    '''
    指定したURLのsoupオブジェクトを取得する
    :param url:
    :return:webページをBSでパースしたもの
    '''
    response = urllib.request.urlopen(url)  # URLからＨＴＭＬ取ってくる
    return BeautifulSoup(response, 'html.parser')

def input_file_name():
    print('input file name. (def = review_text.csv)')
    file_name = input('>>>')
    if file_name == 'def':
        file_name = 'review_text.csv'
    elif '.csv' not in file_name:
        print('error : it is not csv file.')
        sys.exit()
    return file_name

def write_csv(text, file_name, mode):
    with open(file_name, mode) as f:
        writer = csv.writer(f)
        writer.writerow(text)

def read_csv(file_name):
    text = []
    with open(file_name, 'r') as f:
        dataReader = csv.reader(f)
        for t in dataReader:
            text.append(t)
    return text
'''
def review_to_csv(args, num, url):
    num = num // 15
    if len(args) != 3:
        args[1] = 'y'
        n = 2
    else:
        n = int(args[2])
    print('input write review texts file name')
    file_name = input_file_name()
    if os.path.exists(file_name):
        write_csv('', file_name, 'w')
    if file_name == None:
        print('error : file name is None')
        sys.exit()
    for i in range(num):
        soup = get_soup(url)
        (texts, url) = get_pages_info(soup)
        if url == None:
            print("error : URL is None.")
            break
        write_csv(texts, file_name, 'a')
'''
def review_to_csv(args, num, url, file_name):
    num = num // 15
    if len(args) != 3:
        args[1] = 'y'
        n = 2
    else:
        n = int(args[2])
    print(file_name, 'writing')
    for i in range(num):
        soup = get_soup(url)
        (texts, url) = get_pages_info(soup)
        if url == None:
            print("error : URL is None.")
            break
        write_csv(texts, file_name, 'a')

def ranking_detail_to_csv(url, i, th):
    soup = get_soup(url)
    Detail = soup.find_all(class_ = 'Detail')
    j = 0
    for d in Detail:
        count = d.find(class_ = 'count')
        if count == None:
            count = 0
        else:
            count = count.text
            count = re.sub('[人（）]','',count)
            count = int(count)
        if count >= th:
            file_name = 'review_text' + str(i)+ '-' + str(j) + '.csv'
            j += 1
            url_tag = d.find(onclick="onclickcatalyst('カテゴリトップランキング','人気ランキング');")
            urlr = url_tag.attrs['href']
            urlr = re.sub('item', 'review', urlr)
            url =  'http://review.kakaku.com' + urlr
            page_to_csv(url, file_name)

def ranking_scraip(url, th):
    i = 1
    url_o = url
    os.chdir('./reviews')
    soup = get_soup(url)
    dir_name = soup.h3.text
    check_dir(dir_name)
    while True:
        if i == 1:
            ranking_detail_to_csv(url_o, i, th)
        else:
            url = url_o  + str(i) + '/'
            ranking_detail_to_csv(url, i, th)
        i += 1
        if i > 5:
            break
    os.chdir('/home/Hiroto/PD3')

def check_dir(name):
    dir = os.listdir()
    if name in dir:
        os.chdir('./' + name)
    else:
        os.mkdir(name)
        os.chdir('./' + name)


def page_to_csv(url, file_name):
    soup = get_soup(url)
    args = ['temp','y',2]
    num = get_review_count(soup)
    name,rate = get_hed_csv(soup)
    hed = []
    hed.extend([name, rate, num])
    write_csv(hed, file_name, 'a')
    review_to_csv(args, num, url, file_name)

def get_hed_csv(soup):
    return soup.find(itemprop = 'name').text, soup.find(itemprop = 'ratingValue').text

def G_prep_from_csv(args):
    if len(args) != 3:
        args[1] = 'y'
        n = 2
    else:
        n = int(args[2])
    print('input read review texts file name')
    file_name = input_file_name()
    if file_name == None:
        print('error: file_name is None')
        sys.exit()
    texts = read_csv(file_name)
    G = nx.Graph()
    for i in range(1, len(texts)):
        for t in texts[i]:
            (chunk, link, hinshi) = get_kakari(t)
            if args[1] == 'n':
                N_gram_G_input(n, chunk, G)
            elif args[1] == 'y':
                kakariuke_G_input(chunk, link, n, G)
            chunk.clear()
            link.clear()
    return G

def G_prep_from_csv_auto(args, file_name):
    if len(args) != 3:
        args[1] = 'y'
        n = 2
    else:
        n = int(args[2])
    os.chdir('./reviews')
    if file_name == None:
        print('error: file_name is None')
        sys.exit()
    texts = read_csv(file_name)
    G = nx.Graph()
    for i in range(1, len(texts)):
        for t in texts[i]:
            (chunk, link, hinshi) = get_kakari(t)
            if args[1] == 'n':
                N_gram_G_input(n, chunk, G)
            elif args[1] == 'y':
                kakariuke_G_input(chunk, link, n, G)
            chunk.clear()
            link.clear()
    return G

def get_review_count(soup):
    '''
    レビューの数を返す
    :param soup:パースしたHTML
    :return:レビューの総数
    '''
    return int(soup.find(itemprop='reviewCount').text)


def get_review_texts(soup):
    '''
    レビュー文章の部分を返す
    :param soup:パースしたHTML
    :return:文字列でレビュー文章を返す
    '''
    return [review_html.text for review_html in soup.find_all(class_='revEntryCont')]


def next_url(soup):
    """
    次のページのURLを探し出す
    :param soup: パースした HTML
    nextarrow: 【次のページへ】の矢印の部分のHTML
    urlh: URLの頭部分
    :return:次のページのURLを返す
    """
    nextarrow = soup.find(class_='arrowNext01')
    if nextarrow is None:
        return None
    urlh = 'http://review.kakaku.com'
    return urlh + nextarrow.attrs['href']


def get_kakari(text):
    """
    CaboChaによる形態素解析＋係り受け解析
    :param text: レビュー文章
    :return: chunk=文節, link=その文節の係る文節番号, hinshi=品詞　のリスト
    """
    chunk_list = []
    link_list = []
    hinshi_list = []
    pattern = re.compile("(.*)(:)(.*)")
    cabocha = CaboCha.Parser("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    tree = cabocha.parse(text)

    for i in range(tree.chunk_size()):
        chunk = tree.chunk(i)
        link_list.append(chunk.link)

        for j in range(chunk.feature_list_size):
            word = chunk.feature_list(j)
            indexP0 = word.find('P0:')
            tex = pattern.search(word)
            if indexP0 != -1:
                hinshi_list.append(tex.group(3))  # ~P0:というパターンを探し出して品詞だけ格納

        token_feature = tree.token(chunk.token_pos).feature.split(',')
        chunk_list.append(token_feature[6])
        '''
        for ix in range(chunk.token_pos, chunk.token_pos + chunk.token_size):
            token_feature = tree.token(ix).feature.split(',')  # token_featureに、featureの中身をリストにして格納
            # print('token_feature[0] = ', token_feature[0])
            # print('tree.token(ix).surface = ',tree.token(ix).surface)
            chunk_list.append(token_feature[6])
            break
        '''
    return (chunk_list, link_list, hinshi_list)


def kakariuke_G_input(node, edge, n, G: nx.Graph):
    """
    係り受けを考慮したN-gram共起ネットワーク
    :param node: ノードのリスト(単語リスト)
    :param edge: エッジのリスト　
    :param n: N-gramの範囲
    :param G: グラフ
    :return:
    """
    for i in range(len(node)):
        node1 = node[i]
        node2 = node[edge[i]]
        edge_input(node1, node2, G)
        current = edge[edge[i]]
        for j in range(1, n):
            if edge[current] is -1:
                break
            else:
                node2 = node[current]
                edge_input(node1, node2, G)
                current = edge[current]


def edge_input(node1: Any, node2: Any, G: nx.Graph):
    """
    グラフにノードとエッジを追加
    :param node1: ノード１
    :param node2: ノード２
    :param G: グラフ
    :return:
    """
    for n in (node1, node2):
        if n in G:
            G.node[n]['count'] += 1
        else:
            G.add_node(n, count=1)
    if node1 is node2:
        pass
    elif G.has_edge(node1, node2):
        G.edge[node1][node2]['count'] += 1
    else:
        G.add_edge(node1, node2, count=1)


def N_gram_G_input(n, node, G):
    """
    係り受けを考慮しないN-gram共起ネットワーク
    :param n: 関係をつける範囲
    :param node: ノードのリスト
    :param G: グラフ
    :return:
    """
    for i in range(len(node)):
        node1 = node[i]
        for j in range(1, n):
            if i + j < len(node):
                node2 = node[i + j]
                edge_input(node1, node2, G)


def G_weight_gen(G, word_sum_node):
    """
    重みweightを出現頻度に変換
    :param G:グラフ
    :return: なし
    """
    node_sum = 0
    edge_sum = 0
    for v in G.nodes_iter():
        G.node[v]['prob'] = G.node[v]['count'] / word_sum_node
        node_sum += G.node[v]['prob']
    for u, v in G.edges_iter():
        G.edge[u][v]['prob'] = G.edge[u][v]['count'] / word_sum_node
        #G.edge[u][v]['prob'] = G.node[u]['prob'] * G.node[v]['prob']
        edge_sum += G.edge[u][v]['prob']
    G_weight_assert(G, edge_sum, node_sum)
    bad_edges = find_bad_edges(G, 'prob', 'prob')
    if len(bad_edges) > 0:
        print(bad_edges)
        del_unnes_edges(G, bad_edges)
        print('bad_edges deleted')


def G_weight_assert(G, edge_sum, node_sum):
    print('edge_sum = ', edge_sum)
    print('node_sum = ', node_sum)
    assert(node_sum >= 0.99)


def del_roop(G, del_count_max):
    bad_edges = []
    i = 0
    while True:
        bad_edges = find_bad_edges(G, 'prob', 'prob')
        if len(bad_edges) == 0 or i == del_count_max:
            break
        else:
            del_unnes_edges(G, bad_edges)
        bad_edges.clear()
        i += 1


def clique_enum(G):
    c = list(nx.find_cliques(G))
    c_max = nx.graph_clique_number(G)
    for clique in c:
        if len(clique) == c_max:
            print(clique)


def check_G(G, edge_weight_name, node_weight_name):
    """
    データ研磨にかけるうえで不正なグラフではないか調べる
    :param G: グラフ
    :param edge_weight_name: エッジについてる重み属性の名前
    :param node_weight_name: ノードについてる重み属性の名前
    :return:
    """
    for (u, v) in G.edges_iter():
        if G.edge[u][v][edge_weight_name] > min(G.node[u][node_weight_name], G.node[v][node_weight_name]):
            print(G.edge[u][v])
        elif u == v:
            print(u, v)
        assert (G.edge[u][v][edge_weight_name] <= min(G.node[u][node_weight_name], G.node[v][node_weight_name]))
        assert (u != v)


def find_bad_edges(G, node_weight_name, edge_weight_name):
    """
    異常なエッジを探す
    :param G: グラフ
    :param node_weight_name:ノードの重みの名前
    :param edge_weight_name: エッジの重みの名前
    :return: 異常なエッジのリスト
    """
    bad_edges = []
    for (u, v) in G.edges_iter():
        if G.edge[u][v][edge_weight_name] > min(G.node[u][node_weight_name], G.node[v][node_weight_name]):
            bad_edges.append((u, v))
        elif u == v:
            bad_edges.append((u,v))
    return bad_edges

def set_npmi(G, th):
    """
    エッジの重みをNPMIにして、閾値未満のエッジを削除
    :param G: グラフ
    :return: NPMIの数値が閾値未満のエッジのリスト
    """
    del_edges=[]
    for (u, v) in G.edges_iter():
       G.edge[u][v]['npmi'] = npmi(G.edge[u][v]['prob'], G.node[u]['prob'], G.node[v]['prob'])
       if G.edge[u][v]['npmi'] < th:
           del_edges.append((u,v))
    if len(del_edges) > 0:
        del_unnes_edges(G, del_edges)

def add_degree(G):
    """
    ノードに次数を追加して次数の平均を返す。
    :param G:
    :return:
    """
    degree = nx.degree(G)
    for n in G.nodes_iter():
        G.node[n]['degree'] = degree[n]

def get_degree_average(G):
    """
    グラフの次数の平均を返す
    :param G:
    :return: グラフの次数の平均を返す
    """
    return np.average(list(nx.degree(G).values()))

def get_sorted_bett_cent(G):
    """
    媒介中心性の高いノード順にソートしたリストを返す
    :param G: グラフ
    :return: ソート済みのリスト
    """
    return sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)

def sort_nodes_degree_in_clique(G, b_clique):
    """
    クリークの中で次数の高い順にソートしたリストを返す
    :param G:
    :param b_clique: 媒介中心性の高いノードを含むクリークのリスト
    :return: ソートしたリスト
    """
    b_c_list = []
    sorted_list = []
    for a in b_clique:
        for b in a:
            for c in b:
                b_c_list.append([c, G.node[c]['degree']])
        sort_list = sorted(b_c_list, key=lambda x: x[1], reverse=True)
        b_c_list.clear()
        sorted_list.append(sort_list)
    return sorted_list

def get_bett_clique(bett, range):
    """
    媒介中心性の高いノードを含むクリークのリストを返す
    :param bett: 媒介中心性の高い順にソートされたノードのリスト
    :param range: 対象とする範囲
    :return:
    """
    bett_c = []
    f = 0
    for u, v in bett[:range]:
        current = nx.cliques_containing_node(Z, u)
        for bet in bett_c:
            if bet == current:
                f = 1
                break
        if f == 1:
            f = 0
            pass
        else:
            bett_c.append(nx.cliques_containing_node(Z, u))
    return bett_c

def get_central_node(G, th, attr_name):
    """
    中心性が閾値より高いノードのリストを返す
    :param G:
    :param th:　閾値
    :param attr_name: 属性の名前
    :return:
    """
    central_nodes = []
    for n in G.nodes_iter():
        if G.node[n][attr_name] >= th:
            central_nodes.append(n)
    return central_nodes

def del_unnes_edges(G, del_edges):
    """
    不要なエッジを削除
    :param G: グラフ
    :param del_edges:不要なエッジのタプルのリスト
    :return:
    """
    for (u,v) in del_edges:
        G.remove_edge(u,v)
        assert(G.has_edge(u,v) == False)


def url_input():
    """
    URLの入力
    :return:入力されたURL
    """
    url = input('input url :')
    #実験用
    if url == 'def':
        url = 'http://review.kakaku.com/review/K0000903380/#tab'
    return url

def word_sum_node_create(G):
    """
    レビューに出現するワードの総数
    :param G: グラフ
    :return: ワードの総数
    """
    word_sum_node = 0
    for node in G.nodes_iter():
        word_sum_node += G.node[node]['count']
    return word_sum_node

'''
def word_sum_edge_create(G):
    word_sum_edge = 0
    for u,v in G.edges_iter():
        word_sum_edge += G.edge[u][v]['count']
    return word_sum_edge
'''

def get_pages_info(soup):
    """
    レビューと次のページのURLを取得
    :param soup: パースしたHTML
    :return: レビューテキストと次のページのURL
    """
    return (get_review_texts(soup), next_url(soup))


def G_prep_from_url( args, num, url):
    """
    グラフ作成
    :param args:メインの引数
    :param num:全体のレビューページのページ数
    :param url:レビューページのURL
    :return:収集した単語の合計(int)
    """
    G = nx.Graph()
    num = num // 15
    if len(args) != 3:
        args[1] = 'y'
        n = 2
    else:
        n = int(args[2])
    for i in range(num):
        soup = get_soup(url)
        (texts, url) = get_pages_info(soup)
        print(url)
        if url == None:
            print("error : URL is None.")
            break
        for t in texts:
            (chunk, link, hinshi) = get_kakari(t)
            if args[1] == 'n':
                N_gram_G_input(n, chunk, G)
            elif args[1] == 'y':
                kakariuke_G_input(chunk, link, n, G)
            chunk.clear()
            link.clear()
    return G

'''
if __name__ == '__main__':
    url = 'http://review.kakaku.com/review/K0000903380/#tab'
    G = nx.Graph()
    #mecab = MeCab.Tagger("-Ochasen") #ochasen互換のほうが見た目が読みやすいだけ
    n = 0

    soup = get_soup(url)
    num = get_review_count(soup)
    num = num//15
    chunk = []
    link = []
    hinshi = []
    word_sum = 0


    for i in range(num):
        soup = get_soup(url)
        texts = get_review_texts(soup)
        url = next_url(soup)
        if url is None:
            print("error : URL is None.")
            break

        for j in range(len(texts)):
            (chunk, link, hinshi) = get_kakari(texts[j])
            if mode == 'n':
                N_gram_G_input(n, chunk, G)
            elif mode == 'y':
                kakariuke_G_input(chunk, link, n, G)
            word_sum = word_sum + len(chunk)
            chunk.clear()
            link.clear()

    print('word_sum = ',word_sum)
    print("研磨前ノード数=",G.order())
    print("研磨前エッジ数=",G.size())
    G_weight_gen(G , word_sum)
    H = data_polish(G)
    cliques = nx.find_cliques(H)
    print("ノード数=",G.order())
    print("エッジ数=",G.size())
    print('end')
'''