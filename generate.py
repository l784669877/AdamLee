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
from langchain.schema import Document
import json


def add_gen_record(db, item, flow):
    """
    添加新的记录，作为后续few-shot的示例。
    :param db: 当前数据库对象
    :param item: 需要添加的知识数据
    :return: None
    """
    ids = db.add_documents([Document(page_content=item[0],
                               metadata={'answer': item[1], 'flow': flow})])
    return ids

generate_template = '''1. 本项目旨在通过生成式AI生成SCL代码，用于编程、监控和维护西门子自动化系统的综合性自动化工程软件平台TIA Portal。SCL编程是一种高级编程语言，类似于Pascal和C语言，主要用于复杂逻辑控制和数据处理的自动化应用。作为SCL编程代码助手，你需要根据提供的JSON格式的参数需求和代码工作流程说明，生成对应的SCL代码（代码部分放在`scl`标签之间）。
·请求参数包含以下字段，定义SCA函数时，其中结构体成员的变量名称应用双引号括起来：
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


2. 数据类型转换规则：
- 如果需要进行数据类型转换，请仅参考以下规则，不要自行虚构转换函数：
  {switch}

  
3. 语法规则：
- 基本运算规则：
加法：使用 + 符号。Result := Operand1 + Operand2;
减法：使用 - 符号。Result := Operand1 - Operand2;
乘法：使用 * 符号。Result := Operand1 * Operand2;
除法：使用 / 符号。Result := Operand1 / Operand2;
取余（求模）：使用MOD关键字。Result := Operand1 MOD Operand2;
平方：SCL中没有直接的平方运算符，但可以通过乘法实现。SquareResult := Operand1 * Operand1;

- 如果任务描述输入的的json中指定了return_value，则需要返回结果。
SCL FUNCTION最终输出的示例：
FUNCTION "DTLToString_ISO" : String
{{ S7_Optimized_Access := 'TRUE' }}
...
#DTLToString_ISO := #tempString;  // 返回FUNCTION最终结果

FUNCTION "RandomRange_DInt" : DInt
{{ S7_Optimized_Access := 'TRUE' }}
...             
#RandomRange_DInt := REAL_TO_DINT((#tempNormReal * DINT_TO_REAL(#maxValue - #minValue) + DINT_TO_REAL(#minValue)));  // 返回FUNCTION最终结果
                 
FUNCTION "StringToTaddr" : TADDR_Param
...                 
REGION Process octests 1-4
  ...               
  FOR #tempOctetIndex := 1 TO #NUMBER_OF_IP_OCTETS BY 1 DO
    REGION Octet conversion
      // check if IP Octet contains more then three digits --> Error
      IF #tempCharPosition > #MAX_CHAR_FOR_IP_OCTET_NUMBER THEN
        #error := TRUE;
        #status := #ERR_OCTET_WRONG_NUMBER_OF_CHAR OR INT_TO_WORD(#tempOctetIndex);
        #StringToTaddr := #tempIpAdressTaddr;  // 返回FUNCTION最终结果
        RETURN;
      END_IF;
      // extract octet string
      #tempOctetString := LEFT(IN := #tempAddressString, L := #tempCharPosition - 1);
      // Check if Octet string is empty --> Error
      IF LEN(#tempOctetString) = #EMPTY_STRING THEN
        #error := TRUE;
        #status := #ERR_OCTET_STRING_IS_EMPTY OR INT_TO_WORD(#tempOctetIndex);
        #StringToTaddr := #tempIpAdressTaddr;  // 返回FUNCTION最终结果
        RETURN;
      END_IF;
      // check if the octet number exeeds the maximum possible range of 255 of USInt --> Error
      IF #tempNumber > #MAX_IP_ADDRESS_OCTET_NUMBER THEN
        #error := TRUE;
        #status := #ERR_OCTET_EXCEEDS_MAX_IP_ADDRESS OR INT_TO_WORD(#tempOctetIndex);
        #StringToTaddr := #tempIpAdressTaddr;  // 返回FUNCTION最终结果
        RETURN;
      END_IF;
  ...               
END_REGION Process octests 1-4           
在定义有返回值的SCL函数的返回逻辑时，只需关注与函数名同名的变量，而不需要额外处理状态或错误变量。如果任务描述的json中没有定义"output"我不需要输出参数有error和status用于指示函数是否成功执行，不要擅自添加，只需要遵从"return_value"中对函数返回值的要求。如果任务描述的json中有"output"则需要再定义块接口的输出参数。

- RETURN语句用于从函数（Function）或功能块（Function Block）中退出，并返回到调用它的地方。RETURN 语句可以用于在满足特定条件时提前结束函数的执行，或者在函数执行完毕后正常返回。
在函数或功能块的执行过程中，如果某个条件满足，可以使用 RETURN 语句提前退出函数，不再执行后续的代码。
这对于处理错误情况或优化代码执行流程非常有用。
在某些情况下，RETURN 语句可以省略，因为函数或功能块在执行完所有代码后会自动返回。
语法：RETURN 语句不需要任何参数。
示例：
IF #errorCondition THEN
    RETURN;
END_IF;

- 本编程语言不支持“..”操作符和“^”操作符。

- 在PLC编程中，特别是在使用SCL（Structured Control Language）时，VAR CONSTANT块用于定义常量，而数组数据类型通常不能定义为常量。正确的做法是将数组定义在VAR块中。
如下的定义方式会报错：
VAR CONSTANT
   A : Array[1..5] of Int := [1, 1, 1, 1, 1];
END_VAR
注意！正确的写法应为：
VAR
   // 错误写法
   A : Array[1..5] of Int := [1, 1, 1, 1, 1];
END_VAR
如下的定义方式会报错：
VAR CONSTANT
    B : Array[1..3] of Bool := [TRUE, FALSE, TRUE];
END_VAR
注意！正确的写法应为：
VAR
    B : Array[1..3] of Bool := [TRUE, FALSE, TRUE];
END_VAR


- 当前使用的CPU或库版本不支持块“reset”。

- 定义块变量必须用双引号将变量名括起来，例如"date" : DTL; 后续将用到#date.YEAR...

- 没有`INT_TO_STRG`内置函数，请勿使用。

- 数组与单独变量不同，例如`#Byte[0]`和`#Byte0`是不同的。为了代码健壮性，如果涉及多个类似`#Byte0`、`#Byte1`的变量，应将这些变量放入一个数组中，并在变量块中预先定义数组。

- 使用定时器`AutoModeTimer`时，必须为所有定时器设置预设时间（PT），例如：`#AutoModeTimer(IN := TRUE, PT := T#1s);`。
- 字符串的最大长度为254。

- “String”和“Char”数据类型不兼容，不能直接相加。可通过直接修改字符串中的特定位置来操作字符串和字符。

- 在变量定义中，双引号用于db块的符号，不带双引号的是db块中的变量。例如：
  ```scl
  VAR_INPUT "date" : DTL;
  separator : Char;
  END_VAR
  ```这里的`date`需要使用双引号。

  - 无法将数据类型“Variant”隐式转换为数据类型“String”。例如：
VAR_IN_OUT 
  searchIn : Variant;
END_VAR
VAR_TEMP 
  tempString : String;
END_VAR
...
#tempString := #searchIn;的写法是错误的

- dataBase := [0] * MAX_DATABASE_SIZE的写法是不对的。

- 运算符“+” 与“Int” 和“Bool” 的数据类型不兼容。


4. 内置函数：
·注意！使用内置函数时，请仅使用下方列出函数!不要自己编造和虚构下文未提及的内置函数！
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

函数名称：TRUNC
参数：
- <表达式>：Input，浮点数，输入值。
- _<数据类型>：整数、浮点数，函数值的数据类型，默认为DINT。
功能描述：直接从输入值中截取整数部分作为函数值返回，不包含小数位。
示例：
VAR_TEMP 
  Tag_Value1 : REAL;
  Tag_Result1 : DINT;
END_VAR
"Tag_Result1" := TRUNC("Tag_Value1");

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


5. 示例代码, 请仿照示例生成代码：
- 参考以下正确示例的语法规则、内置函数的使用和算法思路：
  {few_shot}'''



generate_template_sub = ''


def MyParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input[:]:
#         result += f'''- 示例输入：
# 代码要求：{doc.page_content}
# 代码工作流：{doc.metadata['flow']}

# 示例输出：{doc.metadata['answer']}'''
        result += f'''- 示例输入：
代码要求：{doc.page_content}

示例输出：{doc.metadata['answer']}'''
        result += '\n\n'
    return result[:-1]

def KnowlParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input:
        result += f'''- {doc.page_content}'''
        result += '\n\n'
    return result[:-1]


def to_generate(input_json, input_process, llm, k=2):
    # input = f'''代码要求：{input_json}
    #     代码工作流：{input_process}'''
    input = f'''{input_json}'''
    pr = StrOutputParser()
    embedding = ZhipuAIEmbeddings()
    knowl_block = load_db(embedding, r'database/knowl/block')
    knowl_func = load_db(embedding, r'database/knowl/function')
    knowl_grammar = load_db(embedding, r'database/knowl/grammar')
    knowl_switch = load_db(embedding, r'database/knowl/switch')
    

    generate_db = load_db(embedding, r'database/generate/generate_db')
    generate_db_few_shot_retriever = generate_db.as_retriever(search_kwargs={'k': k})

    prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", generate_template),
                ("human", '''**任务描述:**
{input}

**指引:**

1. **数据类型转换规则：**
    - 仅使用系统提示中给定的数据转换规则。
    - 严禁使用未提及的、自行虚构的数据转换函数。

2. **内置函数使用规则：**
    - 使用内置函数极易报错，因此能不用内置函数就不用，不需要担心效率问题。推荐使用基础语法语句编写SCL代码。
    - 仅使用系统提示中给定的内置函数。
    - 严禁使用任何未提及的、自行虚构的内置函数，例如若未提及SIZEOF，则严禁使用。
    - 严格参考给定的内置函数说明，确保所有提到的参数都被输入，不可以省略某个参数而不输入。
    - 要严格保证参数的数据类型输入正确。若不正确则需要进行数据类型转换。

3. **SCL函数编写：**
    - 严格参考给定的语法规则和示例的算法思路来编写SCL函数。
    - 使用内置函数极易报错，因此能不用内置函数就不用，不需要担心效率问题。推荐使用基础语法语句编写SCL代码。

**输出要求：**
- 请输出完整而详细的思考过程，如内置函数的选取；数据类型转换的过程；数据类型转换函数的选取；最后SCL函数结果返回的思考过程等，如果任务描述的json指定了return_value，则需要返回结果。这些思考过程必须有充足的细节，例如函数形参的数据类型是否符合等。最终的SCL代码仅需要生成一次即可。 
- 在定义有返回值的SCL函数的返回逻辑时，只需关注与函数名同名的变量，而不需要额外处理状态或错误变量。如果任务描述的json中没有定义"output"我不需要输出参数有error和status用于指示函数是否成功执行，不要擅自添加，只需要遵从"return_value"中对函数返回值的要求。如果任务描述的json中有"output"则需要再定义块接口的输出参数。
- 请仅提供完整的SCL代码，使用```scl ```标注代码便于我提取。
                 
开始！''')
            ])
    
    json_dict = json.loads(input_json)
    # 处理switch
    params = [item["type"] for item in json_dict["input"]]
    params = ' '.join(list(set(params)))
    switch_raw = knowl_switch.similarity_search(params, k=40)
    switch = KnowlParser(switch_raw)

    # 处理block
    block_raw = knowl_block.similarity_search(input_json, k=5)
    block = KnowlParser(block_raw)

    # 处理语法
    grammar_raw = knowl_grammar.similarity_search(input_json, k=8)
    grammar = KnowlParser(grammar_raw)

    # 处理函数库
    # func = []
    # # processes = input_process.split('\n\n')
    # # for i, subproc in enumerate(processes):
    # #     subproc_row = knowl_func.similarity_search(subproc, k=3)
    # #     func += subproc_row
    # func += knowl_func.similarity_search(input_json, k=5)
    # func = ['- '+item.page_content + '\n所有给定的参数必须都要显式输入。' for item in func]
    # function = '\n\n'.join(func)
    # func = [item.page_content for item in func]
    # function = '\n\n'.join(list(set(func)))
    function=''

    
    few_shot_retrieval = RunnableParallel(
        {'few_shot': itemgetter("input")| generate_db_few_shot_retriever | RunnableLambda(MyParser),
         'input': itemgetter("input"),
         'grammar': itemgetter("grammar"),
         'function': itemgetter("function"),
         'block': itemgetter("block"),
         'switch': itemgetter("switch")}
    )


    chain = (
        few_shot_retrieval
        | prompt
        | llm | pr)
    
    return (chain.invoke({'input':input, 'grammar':grammar, 'function':function,
                         'block':block, 'switch':switch}), function, switch)




def to_generate_sub(input_json, input_process, llm, k=2):
    return



















if(__name__ == '__main__'):
   zhipuai_api_key="db3b7cdfb3495e9d14f144e43001bc3f.PEmDSwyWyR6ColTL"
   codegeex = ChatOpenAI(
         model_name="codegeex-4",
         openai_api_base="https://open.bigmodel.cn/api/paas/v4",
         openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
         streaming=False,
         verbose=True
      )

   print(to_generate('123', '111', codegeex, 'none' ,k=2))
   
