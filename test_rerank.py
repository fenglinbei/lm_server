#模型下载
from sentence_transformers import CrossEncoder
from BCEmbedding import RerankerModel

model_path = "/WiqunBot/model/bce-reranker-base_v1/"
query = "公园好友列表上限是多少"
passages = [
    '''# 公园优化体验1. 【首页】判断对方是否是好友，是好友点击进入【好友聊天】2. 【我的收藏】点击私聊、私信，判断是否是好友，是好友点击进入【好友聊天】''',
    '''- - - ### 【我的收藏】[[我的收藏.png]]- 排列顺序：在公园内有留言》在公园内没有留言》有留言》没有留言 - 右上角显示“收到的私信”红点提示数量- 显示用户填写的社交信息- 点击详情“弹出”文字简介- ''',
    '''- - -### 【我的人气魅力】[[我的人气魅力.png]]1. 数据来自现场请求2. 最新请求优先排前-然后按照对话条数多的排前-统计双方发送的信息条数,好友列表上限是200'''
]

sentence_pairs = [[query, passage] for passage in passages]
# # init reranker model
# model = CrossEncoder(model_path, max_length=512)

# # calculate scores of sentence pairs
# scores = model.predict(sentence_pairs)
# print(scores)

model = RerankerModel(model_name_or_path=model_path)

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)

print(scores)
print(rerank_results)