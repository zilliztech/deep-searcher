import logging
import os
import glob

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

config = Configuration()  # Customize your config here
config.set_provider_config("llm", "JiekouAI", {"model": "claude-sonnet-4-5-20250929"})
config.set_provider_config("embedding", "JiekouAIEmbedding", {"model": "qwen/qwen3-embedding-8b"})
init_config(config=config)


# You should clone the milvus docs repo to your local machine first, execute:
# git clone https://github.com/milvus-io/milvus-docs.git
# Then replace the path below with the path to the milvus-docs repo on your local machine
# import glob
# all_md_files = glob.glob('xxx/milvus-docs/site/en/**/*.md', recursive=True)
# load_from_local_files(paths_or_directory=all_md_files, collection_name="milvus_docs", collection_description="All Milvus Documents")

# Hint: You can also load a single file, please execute it in the root directory of the deep searcher project
load_from_local_files(
    paths_or_directory='/Users/jason/Documents/work/PPLabs/Platform/deep-searcher/examples/data/car_company_data.pdf',
    collection_name="car_company_data",
    collection_description="car_company_data",
    force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

question = """请从财务和宏观经济角度，对 A 股新能源汽车行业以及行业内 TOP5 车企的发展进行分析。在财务分析部分，需涵盖基本的财务关键指标。
在宏观经济分析部分，考虑宏观经济指标对行业的影响，对 A 股新能源汽车行业整体发展趋势进行总结，并基于财务和宏观经济分析，对 TOP5 新能源车企的未来发展潜力和竞争态势做出比较和预测，指出各车企的优势与挑战。请以专业、严谨的语言，结合具体数据进行分析阐述。
"""

_, _, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")
