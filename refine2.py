from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#openai
import os
from langchain_openai import ChatOpenAI
from utils import SeeWhat
from langchain_core.output_parsers import StrOutputParser
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
import json
from utils import list_json_files
from langchain.schema import Document

def add_refine_record(db, item):
    """
    添加新的记录，作为后续few-shot的示例。
    :param db: 当前数据库对象
    :param item: 需要添加的知识数据
    :return: None
    """
    ids = db.add_documents([Document(page_content=item[0],
                               metadata={'answer': item[1]})])
    return ids


def MyParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input[:]:
        result += f'''示例输入：
代码要求：{doc.page_content}

示例输出：{doc.metadata['answer']}'''
        result += '\n\n\n'
    return result[:-1]



refine_template = '''1. 作为SCL编程代码审查助手，你的任务是根据给定的函数要求，审查由大语言模型生成的代码。专注于审查代码的算法逻辑，并确保它满足要求。

2. 在这个系统中，井号（#）用于引用变量，请在审查过程中保留这一符号。记住，数组或列表的索引通常从0开始。

3. 由于审查过的代码将用于自动化测试，请确保输出的代码完整、可以直接运行。请将SCL代码放在```scl 和``` 标记之间，以便系统可以直接提取和测试。

4. 内置函数：
·注意！使用内置函数时，仅可使用下方列出函数!如果出现了其他函数则会报错！
·下方列出的函数的排序和重要性无关。
·注意正确使用函数参数，你有如下可参考的内置函数：
- 函数名称：FOR
参数：
- <执行变量>：只能是SINT、INT、DINT、USINT、UINT、UDINT之一的数据类型，执行循环时会计算其值的操作数。执行变量的数据类型将确定其它参数的数据类型。
- <起始值>：只能是SINT、INT、DINT、USINT、UINT、UDINT之一的数据类型，表达式，在执行变量首次执行循环时，将分配表达式的值。
- <结束值>：只能是SINT、INT、DINT、USINT、UINT、UDINT之一的数据类型，表达式，在运行程序最后一次循环时会定义表达式的值。
- <Increment>：SINT、INT、DINT、USINT、UINT、UDINT，执行变量在每次循环后都会递增（正增量）或递减（负增量）其值的表达式。如果未指定增量，则在每次循环后执行变量的值加1。
所有给定的参数必须都要显式输入。

- 函数名称：CONCAT
参数：
- IN1：Input，STRING或WSTRING，第一个字符串。
- IN2：Input，STRING或WSTRING，第二个字符串。
函数返回值为STRING或WSTRING，合并后的字符串。
功能描述：
将IN1输入参数中的字符串与IN2输入参数中的字符串合并在一起，结果通过OUT输出参数输出。如果生成的字符串长度大于OUT参数中指定的变量长度，则将生成的字符串限制到可用长度。
示例：
VAR_INPUT 
  date : DTL;
  separator : Char;
END_VAR
VAR_TEMP 
  tempYearString : String[4];
  resultString : String;
END_VAR
...
#resultString := CONCAT(IN1 := #tempYearString, IN2 := #separator);
#resultString := CONCAT(IN1 := string_1, IN2 := string_2, IN3 := string_3, ...)（最多32个输入）
所有给定的参数必须都要显式输入。

函数名称：FIND
参数：
- IN1：Input，STRING或WSTRING，被搜索的字符串。
- IN2：Input，STRING或WSTRING，要搜索的字符串。
函数返回值，INT，字符位置。
功能描述：FIND函数用于在IN1输入参数中的字符串内搜索特定的字符串。该函数使用IN2输入参数指定要搜索的值，搜索从左向右进行。OUT输出参数中输出第一次出现该值的位置。如果搜索返回没有匹配项，则OUT输出参数中将输出值“0”。
示例：
VAR_TEMP 
  tempAddressString : String;
END_VAR
  CHAR_DOT : Char := '.';
END_VAR
#tempCharPosition := FIND(IN1 := #tempAddressString, IN2 := #CHAR_DOT);

函数名称：LEFT
参数：
- IN：Input，STRING或WSTRING，源字符串。
- L：Input，BYTE、INT、SINT、USINT，要提取的字符数。
函数返回值：STRING或WSTRING，提取的部分字符串。
说明：
使用指令“LEFT”提取以 IN 输入参数中字符串的第一个字符开头的部分字符串。在 L 参数中指定要提取的字符数。提取的字符以 (W)STRING 格式通过 OUT 输出参数输出。
示例：
VAR_TEMP 
  tempNumElements : UDInt;
  tempPosInArray : DInt;
  tempLenTextBefore : Int;
  tempPosTextBefore : DInt;
  tempLenTextAfter : Int;
  tempPosTextAfter : Int;
  tempString : String;
END_VAR
...
#extractedString := LEFT(IN := #tempString, L := #tempPosTextAfter - 1);

- 函数名称：VAL_STRG
参数：
- IN：Input，与OUT参数相同的数据类型，需要转换的值。
- SIZE：Input，USINT，写入字符串中字符的数量。
- PREC：Input，USINT，小数点后的位数。
- FORMAT：Input，USINT，控制数字格式。
- P：Input，USINT，字符串中开始写入的位置。
- OUT：Output，STRING，转换结果。
功能描述：
VAL_STRG函数用于将数值转换成字符串。在转换过程中，根据参数P的值确定写入位置，根据参数SIZE确定写入字符的数量，根据参数PREC确定小数点后的位数，根据参数FORMAT确定数字的格式。
示例：
VAR_INPUT 
  "date"  : DTL;
END_VAR
VAR_TEMP 
  tempString : String;
END_VAR
VAR CONSTANT 
  CONVERT_SIZE_MONTH_DAY_HOUR_MINUTE_SECOND : USInt := 2;
  CONVERT_PRECISION : USInt := 0;
  CONVERT_FORMAT_TO_STRING : Word := 16#0000;
  CONVERT_START_POSITION_MONTH : UInt := 6;
END_VAR
...
VAL_STRG(FORMAT := #CONVERT_FORMAT_TO_STRING,
          IN     := #date.MONTH,
          P      := #CONVERT_START_POSITION_MONTH,
          PREC   := #CONVERT_PRECISION,
          SIZE   := #CONVERT_SIZE_MONTH_DAY_HOUR_MINUTE_SECOND,
          OUT    => #tempString);     // DTL MONTH --> String --> #tempPosMonth
所有给定的参数必须都要显式输入。

- 函数名称：Chars_TO_Strg
参数：
- CHARS：Input，VARIANT，复制操作的源，从Array of (W)CHAR / BYTE / WORD开始复制字符。
- PCHARS：Input，DINT，Array of (W)CHAR / BYTE / WORD中的位置，从该位置处开始复制字符。
- CNT：Input，UINT，要复制的字符数。使用值“0”将复制所有字符。
- STRG：Output，STRING或WSTRING，复制操作的目标，(W)STRING数据类型的字符串。
功能描述：Strg_TO_Chars函数用于将字符串中的字符复制到指定的Array结构中。Chars_TO_Strg函数用于将Array中的字符复制到字符串中。两个函数都支持ASCII字符的复制，并能够处理字符数组与字符串之间的数据转换。
示例：
VAR_IN_OUT 
  searchIn : Variant;
END_VAR
VAR_TEMP 
  tempNumElements : UDInt;
  tempPosInArray : DInt;
  tempLenTextBefore : Int;
  tempPosTextBefore : DInt;
  tempLenTextAfter : Int;
  tempPosTextAfter : Int;
  tempString : String;
END_VAR
VAR CONSTANT 
  LEN_STRING : UInt := 254;
  STATUS_TEXT_FOUND : Word := 16#0000;
  WARNING_ONLY_START : Word := 16#9001;
  WARNING_NOTHING_FOUND : Word := 16#9002;
  STATUS_NO_ERROR : Word := 16#0000;
  ERR_NO_ARRAY : Word := 16#8200;
END_VAR
...
Chars_TO_Strg(Chars  := #searchIn,
              pChars := #tempPosInArray, // Subtract offset since pChars is zero based
              Cnt    := UDINT_TO_UINT(MIN(IN1 := #LEN_STRING, IN2 := #tempNumElements)),
              Strg   => #tempString);

- 函数名称：CountOfElements
参数：
- ARRAY：Input，VARIANT，指向要获取元素个数的ARRAY。
函数的返回值：
INT，ARRAY元素的个数或错误信息。
功能描述：
CountOfElements函数用于查询VARIANT指针所包含的ARRAY元素数量。如果是一维ARRAY，则输出ARRAY元素的个数。如果是多维ARRAY，则输出所有维的数量。
错误处理：
如果VARIANT指针不指向ARRAY，或者数据块中的ARRAY被写保护，RET_VAL将返回"0"。
示例：
VAR_IN_OUT 
  item : Variant;
  buffer : Variant;
END_VAR
...
#tempBufferSize := CountOfElements(#buffer);
所有给定的参数必须都要显式输入。

- 函数名称：TypeOf
参数：
- <操作数>：Input，二进制数、整数、浮点数、时间、日期和时间、字符串、VARIANT、ResolvedSymbol，用于查询的操作数。
功能描述：TypeOf函数用于检查VARIANT或ResolvedSymbol变量所指向的变量的数据类型，可以比较块接口中声明的数据类型与其它变量的数据类型或一个直接数据类型，以确定它们是“相等”还是“不相等”。比较操作数可以是基本数据类型或PLC数据类型。该指令只能在IF或CASE指令中使用。
示例：
VAR_IN_OUT 
  item : Variant;
  buffer : Variant;
END_VAR
...
IF (TypeOf(#item) <> TypeOfElements(#buffer)) THEN
    #error := true;
    #status := #ERR_WRONG_TYPE_ITEM;
    
    RETURN;
END_IF;

- 函数名称：TypeOfElements
参数：
- <操作数>：Input，VARIANT，用于查询的操作数
功能描述：检查VARIANT变量的ARRAY元素的数据类型，用于比较VARIANT变量的ARRAY元素的数据类型是否相等
使用场景：在IF或CASE指令中比较VARIANT变量的ARRAY元素的数据类型
示例：
VAR_IN_OUT 
  searchIn : Variant;
END_VAR
...
AND ((TypeOfElements(#searchIn) = Char) OR (TypeOfElements(#searchIn) = Byte))

- 函数名称：IS_ARRAY
参数：
- <操作数>：Input，VARIANT，为ARRAY查询的操作数
功能描述：检查VARIANT是否指向ARRAY数据类型的变量
使用场景：在IF指令中检查VARIANT是否指向ARRAY数据类型的变量
示例：
VAR_IN_OUT 
  searchIn : Variant;
END_VAR
...
IS_ARRAY(#searchIn)

函数名称：MOVE_BLK_VARIANT
参数：
- SRC：Input，VARIANT（指向一个 ARRAY 或一个 ARRAY 元素），ARRAY of <数据类型>，L（可在块接口的“Input”、“InOut”和“Temp”部分进行声明），待复制的源块
- COUNT：Input，UDINT，I、Q、M、D、L，已复制的元素数目。如果参数 SRC 或参数 DEST 中未指定任何 ARRAY，则将参数 COUNT 的值设置为“1”。
- SRC_INDEX：Input，DINT，I、Q、M、D、L，定义要复制的第一个元素：
- SRC_INDEX 参数将从 0 开始计算。如果参数 SRC 中指定了 ARRAY，则参数 SRC_INDEX 中的整数将指定待复制源区域中的第一个元素。而与所声明的 ARRAY 限值无关。如果 SRC 参数中未指定 ARRAY 或者仅指定了 ARRAY 的某个元素，则将 SRC_INDEX 参数的值赋值为“0”。
- DEST_INDEX：Input，DINT，I、Q、M、D、L，定义了目标存储区的起点：DEST_INDEX 参数将从 0 开始计算。如果参数 DEST 中指定了 ARRAY，则参数 DEST_INDEX 中的整数将指定待复制目标范围中的第一个元素。而与所声明的 ARRAY 限值无关。如果参数 DEST 中未指定任何 ARRAY，则将参数 DEST_INDEX 赋值为“0”。
- DEST：Output，VARIANT，L（可在块接口的“Input”、“InOut”和“Temp”部分进行声明），源块中内容将复制到的目标区域。
- 返回值：INT，I、Q、M、D、L，错误信息
功能描述：MOVE_BLK_VARIANT 函数用于将一个存储区（源范围）的数据移动到另一个存储区（目标范围）中。可以将一个完整的 ARRAY 或 ARRAY 的元素复制到另一个相同数据类型的 ARRAY 中。源 ARRAY 和目标 ARRAY 的大小（元素个数）可能会不同。可以复制一个 ARRAY 内的多个或单个元素。要复制的元素数量不得超过所选源范围或目标范围。如果在创建块时使用该指令，则无需确定该 ARRAY，源和目标将使用 VARIANT 进行传输。无论后期如何声明该 ARRAY，参数 SRC_INDEX 和 DEST_INDEX 始终从下限“0”开始计数。如果复制的数据多于可用的数据，则不执行该指令。参数 SRC 的数据类型不能为 BOOL 和 BOOL 型 ARRAY。
示例：
VAR_INPUT 
  initialItem : Variant;
END_VAR
VAR_IN_OUT 
  item : Variant;
  buffer : Variant;
END_VAR
VAR_TEMP 
  tempCounter : Int;
END_VAR
VAR CONSTANT 
  INDEX_BEGINNING : Int := 0;
  COUNT_ELEMENTS : UDInt := 1;
END_VAR
...
#tempInternalError := MOVE_BLK_VARIANT(SRC := #initialItem,
                                      COUNT := #COUNT_ELEMENTS,
                                      SRC_INDEX := #INDEX_BEGINNING,
                                      DEST_INDEX := #tempCounter,
                                      DEST => #buffer);

- 函数名称：UINT_TO_INT
参数：
- <操作数>：Input，UINT待转换的变量
示例：
VAR CONSTANT 
  LEN_STRING : UInt := 254;
END_VAR
UINT_TO_INT(#LEN_STRING)

- 函数名称：INT_TO_WORD
参数：
- <操作数>：Input，INT待转换的变量
示例：
VAR_OUTPUT 
  error : Bool;
  status : Word;
END_VAR
VAR_TEMP 
  tempOctetIndex : Int;
END_VAR
...
#status := #ERR_OCTET_STRING_IS_EMPTY OR INT_TO_WORD(#tempOctetIndex);

- 函数名称：RESET_TIMER
参数：
<IEC 定时器>：Output，IEC_TIMER、TP_TIME、TON_TIME、TOF_TIME、TONR_TIME、IEC_LTIME、TP_LTIME、TON_LTIME、TOF_LTIME、TONR_LTIME，指定要复位的 IEC 定时器。
说明：
使用“复位定时器”指令，可将 IEC 定时器复位为“0”。将指定数据块中定时器的结构组件复位为“0”。
示例：
VAR 
  instTofTimePause  : TOF_TIME;
END_VAR
RESET_TIMER(TIMER := #instTofTimePause);

- 函数名称：REAL_TO_UDINT
参数：
- <操作数>：Input，REAL待转换的变量
示例：
VAR_INPUT 
  frequency : Real := 0.0;
END_VAR
VAR_TEMP 
  tempPulseRate : Real;
END_VAR
VAR CONSTANT 
  SECOND_IN_MS : Real := 1000.0;
END_VAR
...
#statTimePulse := UDINT_TO_TIME(REAL_TO_UDINT((#SECOND_IN_MS * #tempPulseRate / #frequency)));

- 函数名称：UDINT_TO_DWORD
参数：
- <操作数>：Input，UDINT待转换的变量
示例：
VAR_TEMP 
  tempRisingResult : DWord;
  tempNoRisingBits : DWord;
END_VAR
...
#tempNoRisingBits := UDINT_TO_DWORD(DWORD_TO_UDINT(#tempNoRisingBits) - DWORD_TO_UDINT(SHR(IN := #tempNoRisingBits, N := 1) AND 16#55555555));

- 函数名称：DWORD_TO_UDINT
参数：
- <操作数>：Input，UDINT待转换的变量
示例：
VAR_TEMP 
  tempRisingResult : DWord;
  tempNoRisingBits : DWord;
END_VAR
...
#tempNoRisingBits := UDINT_TO_DWORD(DWORD_TO_UDINT(#tempNoRisingBits) - DWORD_TO_UDINT(SHR(IN := #tempNoRisingBits, N := 1) AND 16#55555555));

- 函数名称：DWORD_TO_USINT
参数：
- <操作数>：Input，DWORD待转换的变量
示例：
VAR_OUTPUT 
  noOfFallingBits  : USInt;
END_VAR
VAR_TEMP 
  tempNoFallingBits : DWord;
END_VAR
...
#noOfFallingBits := DWORD_TO_USINT(#tempNoFallingBits);

- 函数名称：SHR
参数：
- IN：Input，待移位的操作数。
- N：Input，整数，指定应将特定值移位的位数。
功能描述：SHR函数用于将参数IN的内容逐位向右移动，并将结果作为函数值返回。如果参数N的值为“0”，则将参数IN的值作为结果。如果参数N的值大于可用位数，则参数IN的值将向右移动该位数个位置。无符号值移位时，用零填充操作数左侧区域中空出的位。如果指定值有符号，则用符号位的信号状态填充空出的位。
示例：
VAR_TEMP 
  tempRisingResult : DWord;
  tempNoRisingBits : DWord;
END_VAR
...
#tempNoRisingBits := UDINT_TO_DWORD(DWORD_TO_UDINT(#tempNoRisingBits) - DWORD_TO_UDINT(SHR(IN := #tempNoRisingBits, N := 1) AND 16#55555555));

- 函数名称：SHL
参数：
- IN：Input，位字符串、整数，要移位的值
- N：Input，USINT、UINT、UDINT，对值(IN)进行移位的位数
- 函数值：位字符串、整数，指令的结果
功能描述：SHL函数用于将参数IN的内容逐位向左移动，并将结果作为函数值返回。
示例：
VAR_TEMP 
  tempRisingResult : DWord;
  tempNoRisingBits : DWord;
END_VAR
...
#tempNoRisingBits := UDINT_TO_DWORD(DWORD_TO_UDINT(#tempNoRisingBits) - DWORD_TO_UDINT(SHL(IN := #tempNoRisingBits, N := 1) AND 16#55555555));

函数名称：RD_SYS_T
参数：
函数的返回值：Return，INT，指令的状态
- OUT：Output，DT或DTL或LDT，CPU的日期和时间
功能描述：RD_SYS_T函数用于读取CPU时钟的当前日期和时间（模块时间），并在OUT输出参数中输出读取的日期和时间。得出的值不包含有关本地时区或夏令时的信息。
错误处理：RET_VAL参数用于返回错误信息。例如，错误代码"8081"表示由于数据读取超出OUT参数已选数据类型所允许的范围，因此无法保存。
示例：
VAR_TEMP 
  tempSysTime  : DTL;
  tempTimeDiffrence : LReal;
  tempCalculation : LReal;
  tempRetval : Word;
END_VAR
...
#tempRetval := INT_TO_WORD(RD_SYS_T(OUT => #tempSysTime));

- 函数名称：LOWER_BOUND
参数：
ARR：Input，ARRAY[*]，待读取可变下限的 ARRAY。
DIM：Input，UDINT，待读取可变下限的 ARRAY 维度。
函数值：DINT，结果。
说明：
在函数块或函数的块接口中，可声明 ARRAY[*] 数据类型的变量。这些局部变量可读取 ARRAY 限值。此时，需要在 DIM 参数中指定维数。
示例：
VAR_INPUT 
  matrix1 : Array[*, *] of LReal;
  matrix2 : Array[*, *] of LReal;
END_VAR
VAR_TEMP 
  tempMatrix1LowerBoundRows : DInt;
END_VAR
VAR CONSTANT 
  ROWS : UInt := 1;
END_VAR
...
#tempMatrix1LowerBoundRows := LOWER_BOUND(ARR := #matrix1, DIM := #ROWS);

函数名称：UPPER_BOUND
参数：
ARR：Input，ARRAY[*]，待读取可变上限的 ARRAY。
DIM：Input，UDINT，待读取可变上限的 ARRAY 维度。
函数值：DINT，结果。
说明：
在函数块或函数的块接口中，可声明 ARRAY[*] 数据类型的变量。这些局部变量可读取 ARRAY 限值。此时，需要在 DIM 参数中指定维数。
示例：
VAR_INPUT 
  values : Array[*] of DInt;
END_VAR
VAR_TEMP 
  tempArrayUpperBound : DInt;
END_VAR
VAR CONSTANT 
  DIMENSION_ONE : UInt := 1;
END_VAR
...
#tempArrayUpperBound := UPPER_BOUND(ARR := #values, DIM := #DIMENSION_ONE);


4. 变量的类型转换的审查，你有如下规则：
{switch}

5. 你有如下示例：
{few_shot}'''


def to_refine2(code, descrp, llm, f, s):
    embedding = ZhipuAIEmbeddings()
    refine_db = load_db(embedding, r'database/refine/refine_db')
    refine_db_few_shot_retriever = refine_db.as_retriever(search_kwargs={'k': 2})



    prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", refine_template),
                ("human", '''- 代码要求：{de}
其中请求参数包含以下字段，定义SCA函数时，其中结构体成员的变量名称应用双引号括起来：
- title: 题目标题
- description: 题目的文字描述
- type: 块类型（FUNCTION 或 FUNCTION_BLOCK）
- name: 块名称
- input: 块接口的输入参数列表
  - name: 参数名称
  - type: 参数类型
  - description: 参数的文字描述
  - fields (可选): 结构体成员，仅当参数类型为Struct时出现
    - name: 结构体成员名称
    - type: 结构体成员类型
    - description: 结构体成员的文字描述
- output: 块接口的输出参数列表
  - name: 参数名称
  - type: 参数类型
  - description: 参数的文字描述
  - fields (可选): 结构体成员，仅当参数类型为Struct时出现
    - name: 结构体成员名称
    - type: 结构体成员类型
    - description: 结构体成员的文字描述
- in/out: 块接口的输入输出参数列表
  - name: 参数名称
  - type: 参数类型
  - description: 参数的文字描述
  - fields (可选): 结构体成员，仅当参数类型为Struct时出现
    - name: 结构体成员名称
    - type: 结构体成员类型
    - description: 结构体成员的文字描述
- return_value (可选): 函数的返回值，仅当块类型为FUNCTION时出现
  - type: 返回值类型
  - description: 返回值的文字描述
  - fields (可选): 结构体成员，仅当返回值类型为Struct时出现
    - name: 结构体成员名称
    - type: 结构体成员类型
    - description: 结构体成员的文字描述

                                  
- 提供的代码：{code}

请根据上述要求，检查代码的算法思路，并输出审查后的完整SCL代码。即使没有错误，无论是否用到了循环的临时变量i，请在VAR_TEMP或VAR中添加变量i以备不时之需。即使代码没有变化，也要输出完整的代码，不要省略任何部分。请仅针对算法思路进行审查，不需要关注SCL语法和内置函数的使用。

                 
开始！''')
            ])

    pr = StrOutputParser()

    few_shot_retrieval = RunnableParallel(
    {'few_shot': itemgetter("input")| refine_db_few_shot_retriever | RunnableLambda(MyParser),
    'code': itemgetter("code"),
    'de': itemgetter("de"),
    'function': itemgetter("function"),
    'switch': itemgetter("switch")}
    )
    chain = (few_shot_retrieval
        |prompt | llm | pr)

    return chain.invoke({'code': code, 'de': descrp, 'input':f"代码要求：{descrp}\n\n代码：{code}\n\n以上代码有算法思路问题。请你检查一下，写出详细检查的过程。请输出完整代码不要用//省略代码的内容", 'function': f, 'switch':s})




if(__name__ == '__main__'):
    pass


    

