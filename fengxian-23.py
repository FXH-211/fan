from dotenv import load_dotenv
from langchain.ooutput_parsers import ResponseSchema,StructreeOutputParser
from langchain_core.prompts import PromptTemplate

schemas = [
    ResponseSchema(name='risk_level', description='风险等级：高/中/低'),
    ResponseSchema(name='reason', description='风险原因'),
    ResponseSchema(name='suggestion', description='修改建议'),
]
parser=StructureOutputParser.from_response_schemas(schemas)

prompt=PromptTemplate.from_template(
    '你是一名法律顾问，请分析以下合同条款的风险:{clause}\n\n{output_format}'
).partial(output_format=parser.get_format_instructions())
load_dotenv()
llm = ChatOpenAI(
    base_url='https://api.deepseek.com/',
    model='deepseek-reasoner',
    temperature=0.2,
)#提示词的输出是大模型的输入
chain=prompt|llm|parser
clause=input('请输入')
result=chain.invoke({'clause':clause})
print(result)
