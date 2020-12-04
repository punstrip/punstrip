module.exports = {
    "_id": "59cbd86f6fbbcefb67025b72",
    "bin_name": "kill",
    "bytes": "554889e5897dfc8b45fc83c00083f87f7707b801000000eb05b8000000005dc3",
    "compiler": "gcc",
    "hash": "3a0332e47758c4c2b63c8d24c4ef2aa975edd09ef93b5bd2cabe7f6dc6261a4f",
    "linkage": "dynamic",
    "name": "fcn.0x0000000000405a6f",
    "opcode_hash": "1b41c2d9066251d0b60b52a4c2acf868dd45820d2fbac59f5d5fe1e9f2e57759",
    "optimisation": 0,
    "path": "/root/friendly-corpus/bin-stripped/dynamic/gcc/o0/ggdb/kill",
    "size": 32,
    "type": "inferred-nucleus",
    "vaddr": 4217455,
    "vex": {
        "constants": [
            {
                "type": "Ity_I64",
                "value": "0x0000000000000008"
            },
            {
                "type": "Ity_I64",
                "value": "0x0000000000405a73"
            },
            {
                "type": "Ity_I64",
                "value": "0xfffffffffffffffc"
            },
            {
                "type": "Ity_I64",
                "value": "0x0000000000405a76"
            },
            {
                "type": "Ity_I64",
                "value": "0xfffffffffffffffc"
            },
            {
                "type": "Ity_I64",
                "value": "0x0000000000000007"
            },
            {
                "type": "Ity_I64",
                "value": "0x000000000000007f"
            },
            {
                "type": "Ity_I64",
                "value": "0x0000000000405a7f"
            },
            {
                "type": "Ity_I64",
                "value": "0x000000000000007f"
            },
            {
                "type": "Ity_I64",
                "value": "0x0000000000405a88"
            }
        ],
        "expressions": [
            {
                "tag": "Iex_Get",
                "value": "GET:I64(offset=56)"
            },
            {
                "tag": "Iex_Get",
                "value": "GET:I64(offset=48)"
            },
            {
                "tag": "Iex_Binop",
                "value": "Sub64(t11,0x0000000000000008)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t11"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000000008"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t10"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t10"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t0"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t10"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000405a73"
            },
            {
                "tag": "Iex_Binop",
                "value": "Add64(t10,0xfffffffffffffffc)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t10"
            },
            {
                "tag": "Iex_Const",
                "value": "0xfffffffffffffffc"
            },
            {
                "tag": "Iex_Get",
                "value": "GET:I64(offset=72)"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to32(t16)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t16"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t37"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t13"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t15"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000405a76"
            },
            {
                "tag": "Iex_Binop",
                "value": "Add64(t10,0xfffffffffffffffc)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t10"
            },
            {
                "tag": "Iex_Const",
                "value": "0xfffffffffffffffc"
            },
            {
                "tag": "Iex_Load",
                "value": "LDle:I32(t17)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t17"
            },
            {
                "tag": "Iex_Unop",
                "value": "32Uto64(t20)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t20"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t38"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to32(t19)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t19"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t39"
            },
            {
                "tag": "Iex_Unop",
                "value": "32Uto64(t21)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t21"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t40"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t25"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to32(t25)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t25"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t41"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000000007"
            },
            {
                "tag": "Iex_Unop",
                "value": "32Uto64(t26)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t26"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t42"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t28"
            },
            {
                "tag": "Iex_Const",
                "value": "0x000000000000007f"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000405a7f"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to32(0x000000000000007f)"
            },
            {
                "tag": "Iex_Const",
                "value": "0x000000000000007f"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to32(t28)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t28"
            },
            {
                "tag": "Iex_Binop",
                "value": "CmpLE32U(t46,t45)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t46"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t45"
            },
            {
                "tag": "Iex_Unop",
                "value": "1Uto64(t44)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t44"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t43"
            },
            {
                "tag": "Iex_Unop",
                "value": "64to1(t35)"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t35"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t47"
            },
            {
                "tag": "Iex_RdTmp",
                "value": "t30"
            },
            {
                "tag": "Iex_Const",
                "value": "0x0000000000405a88"
            }
        ],
        "jumpkind": "Ijk_Boring",
        "ninstructions": 7,
        "ntemp_vars": 48,
        "operations": [
            "Iop_Sub64",
            "Iop_Add64",
            "Iop_64to32",
            "Iop_Add64",
            "Iop_32Uto64",
            "Iop_64to32",
            "Iop_32Uto64",
            "Iop_64to32",
            "Iop_32Uto64",
            "Iop_64to32",
            "Iop_64to32",
            "Iop_CmpLE32U",
            "Iop_1Uto64",
            "Iop_64to1"
        ],
        "statements": [
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a6f, 1, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t0 = GET:I64(offset=56)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t11 = GET:I64(offset=48)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t10 = Sub64(t11,0x0000000000000008)"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=48) = t10"
            },
            {
                "tag": "Ist_Store",
                "value": "STle(t10) = t0"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a70, 3, 0) ------"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=56) = t10"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=184) = 0x0000000000405a73"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a73, 3, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t13 = Add64(t10,0xfffffffffffffffc)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t16 = GET:I64(offset=72)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t37 = 64to32(t16)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t15 = t37"
            },
            {
                "tag": "Ist_Store",
                "value": "STle(t13) = t15"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=184) = 0x0000000000405a76"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a76, 3, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t17 = Add64(t10,0xfffffffffffffffc)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t20 = LDle:I32(t17)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t38 = 32Uto64(t20)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t19 = t38"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a79, 3, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t39 = 64to32(t19)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t21 = t39"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t40 = 32Uto64(t21)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t25 = t40"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=16) = t25"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a7c, 3, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t41 = 64to32(t25)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t26 = t41"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=144) = 0x0000000000000007"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t42 = 32Uto64(t26)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t28 = t42"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=152) = t28"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=160) = 0x000000000000007f"
            },
            {
                "tag": "Ist_Put",
                "value": "PUT(offset=184) = 0x0000000000405a7f"
            },
            {
                "tag": "Ist_IMark",
                "value": "------ IMark(0x405a7f, 2, 0) ------"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t45 = 64to32(0x000000000000007f)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t46 = 64to32(t28)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t44 = CmpLE32U(t46,t45)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t43 = 1Uto64(t44)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t35 = t43"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t47 = 64to1(t35)"
            },
            {
                "tag": "Ist_WrTmp",
                "value": "t30 = t47"
            },
            {
                "tag": "Ist_Exit",
                "value": "if (t30) { PUT(offset=184) = 0x405a81; Ijk_Boring }"
            }
        ],
        "sum_expressions": {
            "Iex_Binop": 4,
            "Iex_Const": 10,
            "Iex_Get": 3,
            "Iex_Load": 1,
            "Iex_RdTmp": 32,
            "Iex_Unop": 10
        },
        "sum_operations": {
            "Iop_1Uto64": 1,
            "Iop_32Uto64": 3,
            "Iop_64to1": 1,
            "Iop_64to32": 5,
            "Iop_Add64": 2,
            "Iop_CmpLE32U": 1,
            "Iop_Sub64": 1
        },
        "sum_statements": {
            "Ist_Exit": 1,
            "Ist_IMark": 7,
            "Ist_Put": 9,
            "Ist_Store": 2,
            "Ist_WrTmp": 26
        },
        "temp_var_types": [
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I32",
            "Ity_I32",
            "Ity_I32",
            "Ity_I32",
            "Ity_I32",
            "Ity_I32",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I32",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I32",
            "Ity_I32",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I32",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I1",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I64",
            "Ity_I32",
            "Ity_I64",
            "Ity_I32",
            "Ity_I64",
            "Ity_I32",
            "Ity_I64",
            "Ity_I64",
            "Ity_I1",
            "Ity_I32",
            "Ity_I32",
            "Ity_I1"
        ]
    }
}
