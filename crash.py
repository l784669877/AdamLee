from db_utils import load_db, add_know_record, ZhipuAIEmbeddings


embedding = ZhipuAIEmbeddings()
knowl_block = load_db(embedding, r'database\knowl\knowl_block')

print(knowl_block.similarity_search_with_score('1'))

a=1