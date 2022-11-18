/* Capstone Disassembly Engine */
/* By Nguyen Anh Quynh <aquynh@gmail.com>, 2013-2015 */

#ifdef CAPSTONE_HAS_MIPS

#include <stdio.h>	// debug
#include <string.h>

#include "../../utils.h"

#include "MipsMapping.h"

#define GET_INSTRINFO_ENUM
#include "MipsGenInstrInfo.inc"

#ifndef CAPSTONE_DIET
static const name_map reg_name_maps[] = {
	{ MIPS_REG_INVALID, NULL },

	{ MIPS_REG_PC, "pc"},

	//{ MIPS_REG_0, "0"},
	{ MIPS_REG_0, "zero"},
	{ MIPS_REG_1, "at"},
	//{ MIPS_REG_1, "1"},
	{ MIPS_REG_2, "v0"},
	//{ MIPS_REG_2, "2"},
	{ MIPS_REG_3, "v1"},
	//{ MIPS_REG_3, "3"},
	{ MIPS_REG_4, "a0"},
	//{ MIPS_REG_4, "4"},
	{ MIPS_REG_5, "a1"},
	//{ MIPS_REG_5, "5"},
	{ MIPS_REG_6, "a2"},
	//{ MIPS_REG_6, "6"},
	{ MIPS_REG_7, "a3"},
	//{ MIPS_REG_7, "7"},
	{ MIPS_REG_8, "t0"},
	//{ MIPS_REG_8, "8"},
	{ MIPS_REG_9, "t1"},
	//{ MIPS_REG_9, "9"},
	{ MIPS_REG_10, "t2"},
	//{ MIPS_REG_10, "10"},
	{ MIPS_REG_11, "t3"},
	//{ MIPS_REG_11, "11"},
	{ MIPS_REG_12, "t4"},
	//{ MIPS_REG_12, "12"},
	{ MIPS_REG_13, "t5"},
	//{ MIPS_REG_13, "13"},
	{ MIPS_REG_14, "t6"},
	//{ MIPS_REG_14, "14"},
	{ MIPS_REG_15, "t7"},
	//{ MIPS_REG_15, "15"},
	{ MIPS_REG_16, "s0"},
	//{ MIPS_REG_16, "16"},
	{ MIPS_REG_17, "s1"},
	//{ MIPS_REG_17, "17"},
	{ MIPS_REG_18, "s2"},
	//{ MIPS_REG_18, "18"},
	{ MIPS_REG_19, "s3"},
	//{ MIPS_REG_19, "19"},
	{ MIPS_REG_20, "s4"},
	//{ MIPS_REG_20, "20"},
	{ MIPS_REG_21, "s5"},
	//{ MIPS_REG_21, "21"},
	{ MIPS_REG_22, "s6"},
	//{ MIPS_REG_22, "22"},
	{ MIPS_REG_23, "s7"},
	//{ MIPS_REG_23, "23"},
	{ MIPS_REG_24, "t8"},
	//{ MIPS_REG_24, "24"},
	{ MIPS_REG_25, "t9"},
	//{ MIPS_REG_25, "25"},
	{ MIPS_REG_26, "k0"},
	//{ MIPS_REG_26, "26"},
	{ MIPS_REG_27, "k1"},
	//{ MIPS_REG_27, "27"},
	{ MIPS_REG_28, "gp"},
	//{ MIPS_REG_28, "28"},
	{ MIPS_REG_29, "sp"},
	//{ MIPS_REG_29, "29"},
	{ MIPS_REG_30, "fp"},
	//{ MIPS_REG_30, "30"},
	{ MIPS_REG_31, "ra"},
	//{ MIPS_REG_31, "31"},

	{ MIPS_REG_DSPCCOND, "dspccond"},
	{ MIPS_REG_DSPCARRY, "dspcarry"},
	{ MIPS_REG_DSPEFI, "dspefi"},
	{ MIPS_REG_DSPOUTFLAG, "dspoutflag"},
	{ MIPS_REG_DSPOUTFLAG16_19, "dspoutflag16_19"},
	{ MIPS_REG_DSPOUTFLAG20, "dspoutflag20"},
	{ MIPS_REG_DSPOUTFLAG21, "dspoutflag21"},
	{ MIPS_REG_DSPOUTFLAG22, "dspoutflag22"},
	{ MIPS_REG_DSPOUTFLAG23, "dspoutflag23"},
	{ MIPS_REG_DSPPOS, "dsppos"},
	{ MIPS_REG_DSPSCOUNT, "dspscount"},

	{ MIPS_REG_AC0, "ac0"},
	{ MIPS_REG_AC1, "ac1"},
	{ MIPS_REG_AC2, "ac2"},
	{ MIPS_REG_AC3, "ac3"},

	{ MIPS_REG_CC0, "cc0"},
	{ MIPS_REG_CC1, "cc1"},
	{ MIPS_REG_CC2, "cc2"},
	{ MIPS_REG_CC3, "cc3"},
	{ MIPS_REG_CC4, "cc4"},
	{ MIPS_REG_CC5, "cc5"},
	{ MIPS_REG_CC6, "cc6"},
	{ MIPS_REG_CC7, "cc7"},

	{ MIPS_REG_F0, "f0"},
	{ MIPS_REG_F1, "f1"},
	{ MIPS_REG_F2, "f2"},
	{ MIPS_REG_F3, "f3"},
	{ MIPS_REG_F4, "f4"},
	{ MIPS_REG_F5, "f5"},
	{ MIPS_REG_F6, "f6"},
	{ MIPS_REG_F7, "f7"},
	{ MIPS_REG_F8, "f8"},
	{ MIPS_REG_F9, "f9"},
	{ MIPS_REG_F10, "f10"},
	{ MIPS_REG_F11, "f11"},
	{ MIPS_REG_F12, "f12"},
	{ MIPS_REG_F13, "f13"},
	{ MIPS_REG_F14, "f14"},
	{ MIPS_REG_F15, "f15"},
	{ MIPS_REG_F16, "f16"},
	{ MIPS_REG_F17, "f17"},
	{ MIPS_REG_F18, "f18"},
	{ MIPS_REG_F19, "f19"},
	{ MIPS_REG_F20, "f20"},
	{ MIPS_REG_F21, "f21"},
	{ MIPS_REG_F22, "f22"},
	{ MIPS_REG_F23, "f23"},
	{ MIPS_REG_F24, "f24"},
	{ MIPS_REG_F25, "f25"},
	{ MIPS_REG_F26, "f26"},
	{ MIPS_REG_F27, "f27"},
	{ MIPS_REG_F28, "f28"},
	{ MIPS_REG_F29, "f29"},
	{ MIPS_REG_F30, "f30"},
	{ MIPS_REG_F31, "f31"},

	{ MIPS_REG_FCC0, "fcc0"},
	{ MIPS_REG_FCC1, "fcc1"},
	{ MIPS_REG_FCC2, "fcc2"},
	{ MIPS_REG_FCC3, "fcc3"},
	{ MIPS_REG_FCC4, "fcc4"},
	{ MIPS_REG_FCC5, "fcc5"},
	{ MIPS_REG_FCC6, "fcc6"},
	{ MIPS_REG_FCC7, "fcc7"},

	{ MIPS_REG_W0, "w0"},
	{ MIPS_REG_W1, "w1"},
	{ MIPS_REG_W2, "w2"},
	{ MIPS_REG_W3, "w3"},
	{ MIPS_REG_W4, "w4"},
	{ MIPS_REG_W5, "w5"},
	{ MIPS_REG_W6, "w6"},
	{ MIPS_REG_W7, "w7"},
	{ MIPS_REG_W8, "w8"},
	{ MIPS_REG_W9, "w9"},
	{ MIPS_REG_W10, "w10"},
	{ MIPS_REG_W11, "w11"},
	{ MIPS_REG_W12, "w12"},
	{ MIPS_REG_W13, "w13"},
	{ MIPS_REG_W14, "w14"},
	{ MIPS_REG_W15, "w15"},
	{ MIPS_REG_W16, "w16"},
	{ MIPS_REG_W17, "w17"},
	{ MIPS_REG_W18, "w18"},
	{ MIPS_REG_W19, "w19"},
	{ MIPS_REG_W20, "w20"},
	{ MIPS_REG_W21, "w21"},
	{ MIPS_REG_W22, "w22"},
	{ MIPS_REG_W23, "w23"},
	{ MIPS_REG_W24, "w24"},
	{ MIPS_REG_W25, "w25"},
	{ MIPS_REG_W26, "w26"},
	{ MIPS_REG_W27, "w27"},
	{ MIPS_REG_W28, "w28"},
	{ MIPS_REG_W29, "w29"},
	{ MIPS_REG_W30, "w30"},
	{ MIPS_REG_W31, "w31"},

	{ MIPS_REG_HI, "hi"},
	{ MIPS_REG_LO, "lo"},

	{ MIPS_REG_P0, "p0"},
	{ MIPS_REG_P1, "p1"},
	{ MIPS_REG_P2, "p2"},

	{ MIPS_REG_MPL0, "mpl0"},
	{ MIPS_REG_MPL1, "mpl1"},
	{ MIPS_REG_MPL2, "mpl2"},
};
#endif

const char *Mips_reg_name(csh handle, unsigned int reg)
{
#ifndef CAPSTONE_DIET
	if (reg >= ARR_SIZE(reg_name_maps))
		return NULL;

	return reg_name_maps[reg].name;
#else
	return NULL;
#endif
}

static const insn_map insns[] = {
	// dummy item
	{
		0, 0,
#ifndef CAPSTONE_DIET
		{ 0 }, { 0 }, { 0 }, 0, 0
#endif
	},

#include "MipsMappingInsn.inc"
};

// given internal insn id, return public instruction info
void Mips_get_insn_id(cs_struct *h, cs_insn *insn, unsigned int id)
{
	unsigned int i;

	i = insn_find(insns, ARR_SIZE(insns), id, &h->insn_cache);
	if (i != 0) {
		insn->id = insns[i].mapid;

		if (h->detail) {
#ifndef CAPSTONE_DIET
			memcpy(insn->detail->regs_read, insns[i].regs_use, sizeof(insns[i].regs_use));
			insn->detail->regs_read_count = (uint8_t)count_positive(insns[i].regs_use);

			memcpy(insn->detail->regs_write, insns[i].regs_mod, sizeof(insns[i].regs_mod));
			insn->detail->regs_write_count = (uint8_t)count_positive(insns[i].regs_mod);

			memcpy(insn->detail->groups, insns[i].groups, sizeof(insns[i].groups));
			insn->detail->groups_count = (uint8_t)count_positive8(insns[i].groups);

			if (insns[i].branch || insns[i].indirect_branch) {
				// this insn also belongs to JUMP group. add JUMP group
				insn->detail->groups[insn->detail->groups_count] = MIPS_GRP_JUMP;
				insn->detail->groups_count++;
			}
#endif
		}
	}
}

static const name_map insn_name_maps[] = {
	{ MIPS_INS_INVALID, NULL },

	{ MIPS_INS_ABSQ_S, "absq_s" },
	{ MIPS_INS_ADD, "add" },
	{ MIPS_INS_ADDIUPC, "addiupc" },
	{ MIPS_INS_ADDIUR1SP, "addiur1sp" },
	{ MIPS_INS_ADDIUR2, "addiur2" },
	{ MIPS_INS_ADDIUS5, "addius5" },
	{ MIPS_INS_ADDIUSP, "addiusp" },
	{ MIPS_INS_ADDQH, "addqh" },
	{ MIPS_INS_ADDQH_R, "addqh_r" },
	{ MIPS_INS_ADDQ, "addq" },
	{ MIPS_INS_ADDQ_S, "addq_s" },
	{ MIPS_INS_ADDSC, "addsc" },
	{ MIPS_INS_ADDS_A, "adds_a" },
	{ MIPS_INS_ADDS_S, "adds_s" },
	{ MIPS_INS_ADDS_U, "adds_u" },
	{ MIPS_INS_ADDU16, "addu16" },
	{ MIPS_INS_ADDUH, "adduh" },
	{ MIPS_INS_ADDUH_R, "adduh_r" },
	{ MIPS_INS_ADDU, "addu" },
	{ MIPS_INS_ADDU_S, "addu_s" },
	{ MIPS_INS_ADDVI, "addvi" },
	{ MIPS_INS_ADDV, "addv" },
	{ MIPS_INS_ADDWC, "addwc" },
	{ MIPS_INS_ADD_A, "add_a" },
	{ MIPS_INS_ADDI, "addi" },
	{ MIPS_INS_ADDIU, "addiu" },
	{ MIPS_INS_ALIGN, "align" },
	{ MIPS_INS_ALUIPC, "aluipc" },
	{ MIPS_INS_AND, "and" },
	{ MIPS_INS_AND16, "and16" },
	{ MIPS_INS_ANDI16, "andi16" },
	{ MIPS_INS_ANDI, "andi" },
	{ MIPS_INS_APPEND, "append" },
	{ MIPS_INS_ASUB_S, "asub_s" },
	{ MIPS_INS_ASUB_U, "asub_u" },
	{ MIPS_INS_AUI, "aui" },
	{ MIPS_INS_AUIPC, "auipc" },
	{ MIPS_INS_AVER_S, "aver_s" },
	{ MIPS_INS_AVER_U, "aver_u" },
	{ MIPS_INS_AVE_S, "ave_s" },
	{ MIPS_INS_AVE_U, "ave_u" },
	{ MIPS_INS_B16, "b16" },
	{ MIPS_INS_BADDU, "baddu" },
	{ MIPS_INS_BAL, "bal" },
	{ MIPS_INS_BALC, "balc" },
	{ MIPS_INS_BALIGN, "balign" },
	{ MIPS_INS_BBIT0, "bbit0" },
	{ MIPS_INS_BBIT032, "bbit032" },
	{ MIPS_INS_BBIT1, "bbit1" },
	{ MIPS_INS_BBIT132, "bbit132" },
	{ MIPS_INS_BC, "bc" },
	{ MIPS_INS_BC0F, "bc0f" },
	{ MIPS_INS_BC0FL, "bc0fl" },
	{ MIPS_INS_BC0T, "bc0t" },
	{ MIPS_INS_BC0TL, "bc0tl" },
	{ MIPS_INS_BC1EQZ, "bc1eqz" },
	{ MIPS_INS_BC1F, "bc1f" },
	{ MIPS_INS_BC1FL, "bc1fl" },
	{ MIPS_INS_BC1NEZ, "bc1nez" },
	{ MIPS_INS_BC1T, "bc1t" },
	{ MIPS_INS_BC1TL, "bc1tl" },
	{ MIPS_INS_BC2EQZ, "bc2eqz" },
	{ MIPS_INS_BC2F, "bc2f" },
	{ MIPS_INS_BC2FL, "bc2fl" },
	{ MIPS_INS_BC2NEZ, "bc2nez" },
	{ MIPS_INS_BC2T, "bc2t" },
	{ MIPS_INS_BC2TL, "bc2tl" },
	{ MIPS_INS_BC3F, "bc3f" },
	{ MIPS_INS_BC3FL, "bc3fl" },
	{ MIPS_INS_BC3T, "bc3t" },
	{ MIPS_INS_BC3TL, "bc3tl" },
	{ MIPS_INS_BCLRI, "bclri" },
	{ MIPS_INS_BCLR, "bclr" },
	{ MIPS_INS_BEQ, "beq" },
	{ MIPS_INS_BEQC, "beqc" },
	{ MIPS_INS_BEQL, "beql" },
	{ MIPS_INS_BEQZ16, "beqz16" },
	{ MIPS_INS_BEQZALC, "beqzalc" },
	{ MIPS_INS_BEQZC, "beqzc" },
	{ MIPS_INS_BGEC, "bgec" },
	{ MIPS_INS_BGEUC, "bgeuc" },
	{ MIPS_INS_BGEZ, "bgez" },
	{ MIPS_INS_BGEZAL, "bgezal" },
	{ MIPS_INS_BGEZALC, "bgezalc" },
	{ MIPS_INS_BGEZALL, "bgezall" },
	{ MIPS_INS_BGEZALS, "bgezals" },
	{ MIPS_INS_BGEZC, "bgezc" },
	{ MIPS_INS_BGEZL, "bgezl" },
	{ MIPS_INS_BGTZ, "bgtz" },
	{ MIPS_INS_BGTZALC, "bgtzalc" },
	{ MIPS_INS_BGTZC, "bgtzc" },
	{ MIPS_INS_BGTZL, "bgtzl" },
	{ MIPS_INS_BINSLI, "binsli" },
	{ MIPS_INS_BINSL, "binsl" },
	{ MIPS_INS_BINSRI, "binsri" },
	{ MIPS_INS_BINSR, "binsr" },
	{ MIPS_INS_BITREV, "bitrev" },
	{ MIPS_INS_BITSWAP, "bitswap" },
	{ MIPS_INS_BLEZ, "blez" },
	{ MIPS_INS_BLEZALC, "blezalc" },
	{ MIPS_INS_BLEZC, "blezc" },
	{ MIPS_INS_BLEZL, "blezl" },
	{ MIPS_INS_BLTC, "bltc" },
	{ MIPS_INS_BLTUC, "bltuc" },
	{ MIPS_INS_BLTZ, "bltz" },
	{ MIPS_INS_BLTZAL, "bltzal" },
	{ MIPS_INS_BLTZALC, "bltzalc" },
	{ MIPS_INS_BLTZALL, "bltzall" },
	{ MIPS_INS_BLTZALS, "bltzals" },
	{ MIPS_INS_BLTZC, "bltzc" },
	{ MIPS_INS_BLTZL, "bltzl" },
	{ MIPS_INS_BMNZI, "bmnzi" },
	{ MIPS_INS_BMNZ, "bmnz" },
	{ MIPS_INS_BMZI, "bmzi" },
	{ MIPS_INS_BMZ, "bmz" },
	{ MIPS_INS_BNE, "bne" },
	{ MIPS_INS_BNEC, "bnec" },
	{ MIPS_INS_BNEGI, "bnegi" },
	{ MIPS_INS_BNEG, "bneg" },
	{ MIPS_INS_BNEL, "bnel" },
	{ MIPS_INS_BNEZ16, "bnez16" },
	{ MIPS_INS_BNEZALC, "bnezalc" },
	{ MIPS_INS_BNEZC, "bnezc" },
	{ MIPS_INS_BNVC, "bnvc" },
	{ MIPS_INS_BNZ, "bnz" },
	{ MIPS_INS_BOVC, "bovc" },
	{ MIPS_INS_BPOSGE32, "bposge32" },
	{ MIPS_INS_BREAK, "break" },
	{ MIPS_INS_BREAK16, "break16" },
	{ MIPS_INS_BSELI, "bseli" },
	{ MIPS_INS_BSEL, "bsel" },
	{ MIPS_INS_BSETI, "bseti" },
	{ MIPS_INS_BSET, "bset" },
	{ MIPS_INS_BZ, "bz" },
	{ MIPS_INS_BEQZ, "beqz" },
	{ MIPS_INS_B, "b" },
	{ MIPS_INS_BNEZ, "bnez" },
	{ MIPS_INS_BTEQZ, "bteqz" },
	{ MIPS_INS_BTNEZ, "btnez" },
	{ MIPS_INS_CACHE, "cache" },
	{ MIPS_INS_CEIL, "ceil" },
	{ MIPS_INS_CEQI, "ceqi" },
	{ MIPS_INS_CEQ, "ceq" },
	{ MIPS_INS_CFC1, "cfc1" },
	{ MIPS_INS_CFCMSA, "cfcmsa" },
	{ MIPS_INS_CINS, "cins" },
	{ MIPS_INS_CINS32, "cins32" },
	{ MIPS_INS_CLASS, "class" },
	{ MIPS_INS_CLEI_S, "clei_s" },
	{ MIPS_INS_CLEI_U, "clei_u" },
	{ MIPS_INS_CLE_S, "cle_s" },
	{ MIPS_INS_CLE_U, "cle_u" },
	{ MIPS_INS_CLO, "clo" },
	{ MIPS_INS_CLTI_S, "clti_s" },
	{ MIPS_INS_CLTI_U, "clti_u" },
	{ MIPS_INS_CLT_S, "clt_s" },
	{ MIPS_INS_CLT_U, "clt_u" },
	{ MIPS_INS_CLZ, "clz" },
	{ MIPS_INS_CMPGDU, "cmpgdu" },
	{ MIPS_INS_CMPGU, "cmpgu" },
	{ MIPS_INS_CMPU, "cmpu" },
	{ MIPS_INS_CMP, "cmp" },
	{ MIPS_INS_COPY_S, "copy_s" },
	{ MIPS_INS_COPY_U, "copy_u" },
	{ MIPS_INS_CTC1, "ctc1" },
	{ MIPS_INS_CTCMSA, "ctcmsa" },
	{ MIPS_INS_CVT, "cvt" },
	{ MIPS_INS_C, "c" },
	{ MIPS_INS_CMPI, "cmpi" },
	{ MIPS_INS_DADD, "dadd" },
	{ MIPS_INS_DADDI, "daddi" },
	{ MIPS_INS_DADDIU, "daddiu" },
	{ MIPS_INS_DADDU, "daddu" },
	{ MIPS_INS_DAHI, "dahi" },
	{ MIPS_INS_DALIGN, "dalign" },
	{ MIPS_INS_DATI, "dati" },
	{ MIPS_INS_DAUI, "daui" },
	{ MIPS_INS_DBITSWAP, "dbitswap" },
	{ MIPS_INS_DCLO, "dclo" },
	{ MIPS_INS_DCLZ, "dclz" },
	{ MIPS_INS_DDIV, "ddiv" },
	{ MIPS_INS_DDIVU, "ddivu" },
	{ MIPS_INS_DERET, "deret" },
	{ MIPS_INS_DEXT, "dext" },
	{ MIPS_INS_DEXTM, "dextm" },
	{ MIPS_INS_DEXTU, "dextu" },
	{ MIPS_INS_DI, "di" },
	{ MIPS_INS_DINS, "dins" },
	{ MIPS_INS_DINSM, "dinsm" },
	{ MIPS_INS_DINSU, "dinsu" },
	{ MIPS_INS_DIV, "div" },
	{ MIPS_INS_DIVU, "divu" },
	{ MIPS_INS_DIV_S, "div_s" },
	{ MIPS_INS_DIV_U, "div_u" },
	{ MIPS_INS_DLSA, "dlsa" },
	{ MIPS_INS_DMFC0, "dmfc0" },
	{ MIPS_INS_DMFC1, "dmfc1" },
	{ MIPS_INS_DMFC2, "dmfc2" },
	{ MIPS_INS_DMOD, "dmod" },
	{ MIPS_INS_DMODU, "dmodu" },
	{ MIPS_INS_DMTC0, "dmtc0" },
	{ MIPS_INS_DMTC1, "dmtc1" },
	{ MIPS_INS_DMTC2, "dmtc2" },
	{ MIPS_INS_DMUH, "dmuh" },
	{ MIPS_INS_DMUHU, "dmuhu" },
	{ MIPS_INS_DMUL, "dmul" },
	{ MIPS_INS_DMULT, "dmult" },
	{ MIPS_INS_DMULTU, "dmultu" },
	{ MIPS_INS_DMULU, "dmulu" },
	{ MIPS_INS_DOTP_S, "dotp_s" },
	{ MIPS_INS_DOTP_U, "dotp_u" },
	{ MIPS_INS_DPADD_S, "dpadd_s" },
	{ MIPS_INS_DPADD_U, "dpadd_u" },
	{ MIPS_INS_DPAQX_SA, "dpaqx_sa" },
	{ MIPS_INS_DPAQX_S, "dpaqx_s" },
	{ MIPS_INS_DPAQ_SA, "dpaq_sa" },
	{ MIPS_INS_DPAQ_S, "dpaq_s" },
	{ MIPS_INS_DPAU, "dpau" },
	{ MIPS_INS_DPAX, "dpax" },
	{ MIPS_INS_DPA, "dpa" },
	{ MIPS_INS_DPOP, "dpop" },
	{ MIPS_INS_DPSQX_SA, "dpsqx_sa" },
	{ MIPS_INS_DPSQX_S, "dpsqx_s" },
	{ MIPS_INS_DPSQ_SA, "dpsq_sa" },
	{ MIPS_INS_DPSQ_S, "dpsq_s" },
	{ MIPS_INS_DPSUB_S, "dpsub_s" },
	{ MIPS_INS_DPSUB_U, "dpsub_u" },
	{ MIPS_INS_DPSU, "dpsu" },
	{ MIPS_INS_DPSX, "dpsx" },
	{ MIPS_INS_DPS, "dps" },
	{ MIPS_INS_DROTR, "drotr" },
	{ MIPS_INS_DROTR32, "drotr32" },
	{ MIPS_INS_DROTRV, "drotrv" },
	{ MIPS_INS_DSBH, "dsbh" },
	{ MIPS_INS_DSHD, "dshd" },
	{ MIPS_INS_DSLL, "dsll" },
	{ MIPS_INS_DSLL32, "dsll32" },
	{ MIPS_INS_DSLLV, "dsllv" },
	{ MIPS_INS_DSRA, "dsra" },
	{ MIPS_INS_DSRA32, "dsra32" },
	{ MIPS_INS_DSRAV, "dsrav" },
	{ MIPS_INS_DSRL, "dsrl" },
	{ MIPS_INS_DSRL32, "dsrl32" },
	{ MIPS_INS_DSRLV, "dsrlv" },
	{ MIPS_INS_DSUB, "dsub" },
	{ MIPS_INS_DSUBU, "dsubu" },
	{ MIPS_INS_EHB, "ehb" },
	{ MIPS_INS_EI, "ei" },
	{ MIPS_INS_ERET, "eret" },
	{ MIPS_INS_EXT, "ext" },
	{ MIPS_INS_EXTP, "extp" },
	{ MIPS_INS_EXTPDP, "extpdp" },
	{ MIPS_INS_EXTPDPV, "extpdpv" },
	{ MIPS_INS_EXTPV, "extpv" },
	{ MIPS_INS_EXTRV_RS, "extrv_rs" },
	{ MIPS_INS_EXTRV_R, "extrv_r" },
	{ MIPS_INS_EXTRV_S, "extrv_s" },
	{ MIPS_INS_EXTRV, "extrv" },
	{ MIPS_INS_EXTR_RS, "extr_rs" },
	{ MIPS_INS_EXTR_R, "extr_r" },
	{ MIPS_INS_EXTR_S, "extr_s" },
	{ MIPS_INS_EXTR, "extr" },
	{ MIPS_INS_EXTS, "exts" },
	{ MIPS_INS_EXTS32, "exts32" },
	{ MIPS_INS_ABS, "abs" },
	{ MIPS_INS_FADD, "fadd" },
	{ MIPS_INS_FCAF, "fcaf" },
	{ MIPS_INS_FCEQ, "fceq" },
	{ MIPS_INS_FCLASS, "fclass" },
	{ MIPS_INS_FCLE, "fcle" },
	{ MIPS_INS_FCLT, "fclt" },
	{ MIPS_INS_FCNE, "fcne" },
	{ MIPS_INS_FCOR, "fcor" },
	{ MIPS_INS_FCUEQ, "fcueq" },
	{ MIPS_INS_FCULE, "fcule" },
	{ MIPS_INS_FCULT, "fcult" },
	{ MIPS_INS_FCUNE, "fcune" },
	{ MIPS_INS_FCUN, "fcun" },
	{ MIPS_INS_FDIV, "fdiv" },
	{ MIPS_INS_FEXDO, "fexdo" },
	{ MIPS_INS_FEXP2, "fexp2" },
	{ MIPS_INS_FEXUPL, "fexupl" },
	{ MIPS_INS_FEXUPR, "fexupr" },
	{ MIPS_INS_FFINT_S, "ffint_s" },
	{ MIPS_INS_FFINT_U, "ffint_u" },
	{ MIPS_INS_FFQL, "ffql" },
	{ MIPS_INS_FFQR, "ffqr" },
	{ MIPS_INS_FILL, "fill" },
	{ MIPS_INS_FLOG2, "flog2" },
	{ MIPS_INS_FLOOR, "floor" },
	{ MIPS_INS_FMADD, "fmadd" },
	{ MIPS_INS_FMAX_A, "fmax_a" },
	{ MIPS_INS_FMAX, "fmax" },
	{ MIPS_INS_FMIN_A, "fmin_a" },
	{ MIPS_INS_FMIN, "fmin" },
	{ MIPS_INS_MOV, "mov" },
	{ MIPS_INS_FMSUB, "fmsub" },
	{ MIPS_INS_FMUL, "fmul" },
	{ MIPS_INS_MUL, "mul" },
	{ MIPS_INS_NEG, "neg" },
	{ MIPS_INS_FRCP, "frcp" },
	{ MIPS_INS_FRINT, "frint" },
	{ MIPS_INS_FRSQRT, "frsqrt" },
	{ MIPS_INS_FSAF, "fsaf" },
	{ MIPS_INS_FSEQ, "fseq" },
	{ MIPS_INS_FSLE, "fsle" },
	{ MIPS_INS_FSLT, "fslt" },
	{ MIPS_INS_FSNE, "fsne" },
	{ MIPS_INS_FSOR, "fsor" },
	{ MIPS_INS_FSQRT, "fsqrt" },
	{ MIPS_INS_SQRT, "sqrt" },
	{ MIPS_INS_FSUB, "fsub" },
	{ MIPS_INS_SUB, "sub" },
	{ MIPS_INS_FSUEQ, "fsueq" },
	{ MIPS_INS_FSULE, "fsule" },
	{ MIPS_INS_FSULT, "fsult" },
	{ MIPS_INS_FSUNE, "fsune" },
	{ MIPS_INS_FSUN, "fsun" },
	{ MIPS_INS_FTINT_S, "ftint_s" },
	{ MIPS_INS_FTINT_U, "ftint_u" },
	{ MIPS_INS_FTQ, "ftq" },
	{ MIPS_INS_FTRUNC_S, "ftrunc_s" },
	{ MIPS_INS_FTRUNC_U, "ftrunc_u" },
	{ MIPS_INS_HADD_S, "hadd_s" },
	{ MIPS_INS_HADD_U, "hadd_u" },
	{ MIPS_INS_HSUB_S, "hsub_s" },
	{ MIPS_INS_HSUB_U, "hsub_u" },
	{ MIPS_INS_ILVEV, "ilvev" },
	{ MIPS_INS_ILVL, "ilvl" },
	{ MIPS_INS_ILVOD, "ilvod" },
	{ MIPS_INS_ILVR, "ilvr" },
	{ MIPS_INS_INS, "ins" },
	{ MIPS_INS_INSERT, "insert" },
	{ MIPS_INS_INSV, "insv" },
	{ MIPS_INS_INSVE, "insve" },
	{ MIPS_INS_J, "j" },
	{ MIPS_INS_JAL, "jal" },
	{ MIPS_INS_JALR, "jalr" },
	{ MIPS_INS_JALRS16, "jalrs16" },
	{ MIPS_INS_JALRS, "jalrs" },
	{ MIPS_INS_JALS, "jals" },
	{ MIPS_INS_JALX, "jalx" },
	{ MIPS_INS_JIALC, "jialc" },
	{ MIPS_INS_JIC, "jic" },
	{ MIPS_INS_JR, "jr" },
	{ MIPS_INS_JR16, "jr16" },
	{ MIPS_INS_JRADDIUSP, "jraddiusp" },
	{ MIPS_INS_JRC, "jrc" },
	{ MIPS_INS_JALRC, "jalrc" },
	{ MIPS_INS_LB, "lb" },
	{ MIPS_INS_LBU16, "lbu16" },
	{ MIPS_INS_LBUX, "lbux" },
	{ MIPS_INS_LBU, "lbu" },
	{ MIPS_INS_LD, "ld" },
	{ MIPS_INS_LDC1, "ldc1" },
	{ MIPS_INS_LDC2, "ldc2" },
	{ MIPS_INS_LDC3, "ldc3" },
	{ MIPS_INS_LDI, "ldi" },
	{ MIPS_INS_LDL, "ldl" },
	{ MIPS_INS_LDPC, "ldpc" },
	{ MIPS_INS_LDR, "ldr" },
	{ MIPS_INS_LDXC1, "ldxc1" },
	{ MIPS_INS_LH, "lh" },
	{ MIPS_INS_LHU16, "lhu16" },
	{ MIPS_INS_LHX, "lhx" },
	{ MIPS_INS_LHU, "lhu" },
	{ MIPS_INS_LI16, "li16" },
	{ MIPS_INS_LL, "ll" },
	{ MIPS_INS_LLD, "lld" },
	{ MIPS_INS_LSA, "lsa" },
	{ MIPS_INS_LUXC1, "luxc1" },
	{ MIPS_INS_LUI, "lui" },
	{ MIPS_INS_LW, "lw" },
	{ MIPS_INS_LW16, "lw16" },
	{ MIPS_INS_LWC1, "lwc1" },
	{ MIPS_INS_LWC2, "lwc2" },
	{ MIPS_INS_LWC3, "lwc3" },
	{ MIPS_INS_LWL, "lwl" },
	{ MIPS_INS_LWM16, "lwm16" },
	{ MIPS_INS_LWM32, "lwm32" },
	{ MIPS_INS_LWPC, "lwpc" },
	{ MIPS_INS_LWP, "lwp" },
	{ MIPS_INS_LWR, "lwr" },
	{ MIPS_INS_LWUPC, "lwupc" },
	{ MIPS_INS_LWU, "lwu" },
	{ MIPS_INS_LWX, "lwx" },
	{ MIPS_INS_LWXC1, "lwxc1" },
	{ MIPS_INS_LWXS, "lwxs" },
	{ MIPS_INS_LI, "li" },
	{ MIPS_INS_MADD, "madd" },
	{ MIPS_INS_MADDF, "maddf" },
	{ MIPS_INS_MADDR_Q, "maddr_q" },
	{ MIPS_INS_MADDU, "maddu" },
	{ MIPS_INS_MADDV, "maddv" },
	{ MIPS_INS_MADD_Q, "madd_q" },
	{ MIPS_INS_MAQ_SA, "maq_sa" },
	{ MIPS_INS_MAQ_S, "maq_s" },
	{ MIPS_INS_MAXA, "maxa" },
	{ MIPS_INS_MAXI_S, "maxi_s" },
	{ MIPS_INS_MAXI_U, "maxi_u" },
	{ MIPS_INS_MAX_A, "max_a" },
	{ MIPS_INS_MAX, "max" },
	{ MIPS_INS_MAX_S, "max_s" },
	{ MIPS_INS_MAX_U, "max_u" },
	{ MIPS_INS_MFC0, "mfc0" },
	{ MIPS_INS_MFC1, "mfc1" },
	{ MIPS_INS_MFC2, "mfc2" },
	{ MIPS_INS_MFHC1, "mfhc1" },
	{ MIPS_INS_MFHI, "mfhi" },
	{ MIPS_INS_MFLO, "mflo" },
	{ MIPS_INS_MINA, "mina" },
	{ MIPS_INS_MINI_S, "mini_s" },
	{ MIPS_INS_MINI_U, "mini_u" },
	{ MIPS_INS_MIN_A, "min_a" },
	{ MIPS_INS_MIN, "min" },
	{ MIPS_INS_MIN_S, "min_s" },
	{ MIPS_INS_MIN_U, "min_u" },
	{ MIPS_INS_MOD, "mod" },
	{ MIPS_INS_MODSUB, "modsub" },
	{ MIPS_INS_MODU, "modu" },
	{ MIPS_INS_MOD_S, "mod_s" },
	{ MIPS_INS_MOD_U, "mod_u" },
	{ MIPS_INS_MOVE, "move" },
	{ MIPS_INS_MOVEP, "movep" },
	{ MIPS_INS_MOVF, "movf" },
	{ MIPS_INS_MOVN, "movn" },
	{ MIPS_INS_MOVT, "movt" },
	{ MIPS_INS_MOVZ, "movz" },
	{ MIPS_INS_MSUB, "msub" },
	{ MIPS_INS_MSUBF, "msubf" },
	{ MIPS_INS_MSUBR_Q, "msubr_q" },
	{ MIPS_INS_MSUBU, "msubu" },
	{ MIPS_INS_MSUBV, "msubv" },
	{ MIPS_INS_MSUB_Q, "msub_q" },
	{ MIPS_INS_MTC0, "mtc0" },
	{ MIPS_INS_MTC1, "mtc1" },
	{ MIPS_INS_MTC2, "mtc2" },
	{ MIPS_INS_MTHC1, "mthc1" },
	{ MIPS_INS_MTHI, "mthi" },
	{ MIPS_INS_MTHLIP, "mthlip" },
	{ MIPS_INS_MTLO, "mtlo" },
	{ MIPS_INS_MTM0, "mtm0" },
	{ MIPS_INS_MTM1, "mtm1" },
	{ MIPS_INS_MTM2, "mtm2" },
	{ MIPS_INS_MTP0, "mtp0" },
	{ MIPS_INS_MTP1, "mtp1" },
	{ MIPS_INS_MTP2, "mtp2" },
	{ MIPS_INS_MUH, "muh" },
	{ MIPS_INS_MUHU, "muhu" },
	{ MIPS_INS_MULEQ_S, "muleq_s" },
	{ MIPS_INS_MULEU_S, "muleu_s" },
	{ MIPS_INS_MULQ_RS, "mulq_rs" },
	{ MIPS_INS_MULQ_S, "mulq_s" },
	{ MIPS_INS_MULR_Q, "mulr_q" },
	{ MIPS_INS_MULSAQ_S, "mulsaq_s" },
	{ MIPS_INS_MULSA, "mulsa" },
	{ MIPS_INS_MULT, "mult" },
	{ MIPS_INS_MULTU, "multu" },
	{ MIPS_INS_MULU, "mulu" },
	{ MIPS_INS_MULV, "mulv" },
	{ MIPS_INS_MUL_Q, "mul_q" },
	{ MIPS_INS_MUL_S, "mul_s" },
	{ MIPS_INS_NLOC, "nloc" },
	{ MIPS_INS_NLZC, "nlzc" },
	{ MIPS_INS_NMADD, "nmadd" },
	{ MIPS_INS_NMSUB, "nmsub" },
	{ MIPS_INS_NOR, "nor" },
	{ MIPS_INS_NORI, "nori" },
	{ MIPS_INS_NOT16, "not16" },
	{ MIPS_INS_NOT, "not" },
	{ MIPS_INS_OR, "or" },
	{ MIPS_INS_OR16, "or16" },
	{ MIPS_INS_ORI, "ori" },
	{ MIPS_INS_PACKRL, "packrl" },
	{ MIPS_INS_PAUSE, "pause" },
	{ MIPS_INS_PCKEV, "pckev" },
	{ MIPS_INS_PCKOD, "pckod" },
	{ MIPS_INS_PCNT, "pcnt" },
	{ MIPS_INS_PICK, "pick" },
	{ MIPS_INS_POP, "pop" },
	{ MIPS_INS_PRECEQU, "precequ" },
	{ MIPS_INS_PRECEQ, "preceq" },
	{ MIPS_INS_PRECEU, "preceu" },
	{ MIPS_INS_PRECRQU_S, "precrqu_s" },
	{ MIPS_INS_PRECRQ, "precrq" },
	{ MIPS_INS_PRECRQ_RS, "precrq_rs" },
	{ MIPS_INS_PRECR, "precr" },
	{ MIPS_INS_PRECR_SRA, "precr_sra" },
	{ MIPS_INS_PRECR_SRA_R, "precr_sra_r" },
	{ MIPS_INS_PREF, "pref" },
	{ MIPS_INS_PREPEND, "prepend" },
	{ MIPS_INS_RADDU, "raddu" },
	{ MIPS_INS_RDDSP, "rddsp" },
	{ MIPS_INS_RDHWR, "rdhwr" },
	{ MIPS_INS_REPLV, "replv" },
	{ MIPS_INS_REPL, "repl" },
	{ MIPS_INS_RINT, "rint" },
	{ MIPS_INS_ROTR, "rotr" },
	{ MIPS_INS_ROTRV, "rotrv" },
	{ MIPS_INS_ROUND, "round" },
	{ MIPS_INS_SAT_S, "sat_s" },
	{ MIPS_INS_SAT_U, "sat_u" },
	{ MIPS_INS_SB, "sb" },
	{ MIPS_INS_SB16, "sb16" },
	{ MIPS_INS_SC, "sc" },
	{ MIPS_INS_SCD, "scd" },
	{ MIPS_INS_SD, "sd" },
	{ MIPS_INS_SDBBP, "sdbbp" },
	{ MIPS_INS_SDBBP16, "sdbbp16" },
	{ MIPS_INS_SDC1, "sdc1" },
	{ MIPS_INS_SDC2, "sdc2" },
	{ MIPS_INS_SDC3, "sdc3" },
	{ MIPS_INS_SDL, "sdl" },
	{ MIPS_INS_SDR, "sdr" },
	{ MIPS_INS_SDXC1, "sdxc1" },
	{ MIPS_INS_SEB, "seb" },
	{ MIPS_INS_SEH, "seh" },
	{ MIPS_INS_SELEQZ, "seleqz" },
	{ MIPS_INS_SELNEZ, "selnez" },
	{ MIPS_INS_SEL, "sel" },
	{ MIPS_INS_SEQ, "seq" },
	{ MIPS_INS_SEQI, "seqi" },
	{ MIPS_INS_SH, "sh" },
	{ MIPS_INS_SH16, "sh16" },
	{ MIPS_INS_SHF, "shf" },
	{ MIPS_INS_SHILO, "shilo" },
	{ MIPS_INS_SHILOV, "shilov" },
	{ MIPS_INS_SHLLV, "shllv" },
	{ MIPS_INS_SHLLV_S, "shllv_s" },
	{ MIPS_INS_SHLL, "shll" },
	{ MIPS_INS_SHLL_S, "shll_s" },
	{ MIPS_INS_SHRAV, "shrav" },
	{ MIPS_INS_SHRAV_R, "shrav_r" },
	{ MIPS_INS_SHRA, "shra" },
	{ MIPS_INS_SHRA_R, "shra_r" },
	{ MIPS_INS_SHRLV, "shrlv" },
	{ MIPS_INS_SHRL, "shrl" },
	{ MIPS_INS_SLDI, "sldi" },
	{ MIPS_INS_SLD, "sld" },
	{ MIPS_INS_SLL, "sll" },
	{ MIPS_INS_SLL16, "sll16" },
	{ MIPS_INS_SLLI, "slli" },
	{ MIPS_INS_SLLV, "sllv" },
	{ MIPS_INS_SLT, "slt" },
	{ MIPS_INS_SLTI, "slti" },
	{ MIPS_INS_SLTIU, "sltiu" },
	{ MIPS_INS_SLTU, "sltu" },
	{ MIPS_INS_SNE, "sne" },
	{ MIPS_INS_SNEI, "snei" },
	{ MIPS_INS_SPLATI, "splati" },
	{ MIPS_INS_SPLAT, "splat" },
	{ MIPS_INS_SRA, "sra" },
	{ MIPS_INS_SRAI, "srai" },
	{ MIPS_INS_SRARI, "srari" },
	{ MIPS_INS_SRAR, "srar" },
	{ MIPS_INS_SRAV, "srav" },
	{ MIPS_INS_SRL, "srl" },
	{ MIPS_INS_SRL16, "srl16" },
	{ MIPS_INS_SRLI, "srli" },
	{ MIPS_INS_SRLRI, "srlri" },
	{ MIPS_INS_SRLR, "srlr" },
	{ MIPS_INS_SRLV, "srlv" },
	{ MIPS_INS_SSNOP, "ssnop" },
	{ MIPS_INS_ST, "st" },
	{ MIPS_INS_SUBQH, "subqh" },
	{ MIPS_INS_SUBQH_R, "subqh_r" },
	{ MIPS_INS_SUBQ, "subq" },
	{ MIPS_INS_SUBQ_S, "subq_s" },
	{ MIPS_INS_SUBSUS_U, "subsus_u" },
	{ MIPS_INS_SUBSUU_S, "subsuu_s" },
	{ MIPS_INS_SUBS_S, "subs_s" },
	{ MIPS_INS_SUBS_U, "subs_u" },
	{ MIPS_INS_SUBU16, "subu16" },
	{ MIPS_INS_SUBUH, "subuh" },
	{ MIPS_INS_SUBUH_R, "subuh_r" },
	{ MIPS_INS_SUBU, "subu" },
	{ MIPS_INS_SUBU_S, "subu_s" },
	{ MIPS_INS_SUBVI, "subvi" },
	{ MIPS_INS_SUBV, "subv" },
	{ MIPS_INS_SUXC1, "suxc1" },
	{ MIPS_INS_SW, "sw" },
	{ MIPS_INS_SW16, "sw16" },
	{ MIPS_INS_SWC1, "swc1" },
	{ MIPS_INS_SWC2, "swc2" },
	{ MIPS_INS_SWC3, "swc3" },
	{ MIPS_INS_SWL, "swl" },
	{ MIPS_INS_SWM16, "swm16" },
	{ MIPS_INS_SWM32, "swm32" },
	{ MIPS_INS_SWP, "swp" },
	{ MIPS_INS_SWR, "swr" },
	{ MIPS_INS_SWXC1, "swxc1" },
	{ MIPS_INS_SYNC, "sync" },
	{ MIPS_INS_SYNCI, "synci" },
	{ MIPS_INS_SYSCALL, "syscall" },
	{ MIPS_INS_TEQ, "teq" },
	{ MIPS_INS_TEQI, "teqi" },
	{ MIPS_INS_TGE, "tge" },
	{ MIPS_INS_TGEI, "tgei" },
	{ MIPS_INS_TGEIU, "tgeiu" },
	{ MIPS_INS_TGEU, "tgeu" },
	{ MIPS_INS_TLBP, "tlbp" },
	{ MIPS_INS_TLBR, "tlbr" },
	{ MIPS_INS_TLBWI, "tlbwi" },
	{ MIPS_INS_TLBWR, "tlbwr" },
	{ MIPS_INS_TLT, "tlt" },
	{ MIPS_INS_TLTI, "tlti" },
	{ MIPS_INS_TLTIU, "tltiu" },
	{ MIPS_INS_TLTU, "tltu" },
	{ MIPS_INS_TNE, "tne" },
	{ MIPS_INS_TNEI, "tnei" },
	{ MIPS_INS_TRUNC, "trunc" },
	{ MIPS_INS_V3MULU, "v3mulu" },
	{ MIPS_INS_VMM0, "vmm0" },
	{ MIPS_INS_VMULU, "vmulu" },
	{ MIPS_INS_VSHF, "vshf" },
	{ MIPS_INS_WAIT, "wait" },
	{ MIPS_INS_WRDSP, "wrdsp" },
	{ MIPS_INS_WSBH, "wsbh" },
	{ MIPS_INS_XOR, "xor" },
	{ MIPS_INS_XOR16, "xor16" },
	{ MIPS_INS_XORI, "xori" },

	// alias instructions
	{ MIPS_INS_NOP, "nop" },
	{ MIPS_INS_NEGU, "negu" },

	{ MIPS_INS_JALR_HB, "jalr.hb" },
	{ MIPS_INS_JR_HB, "jr.hb" },
};

const char *Mips_insn_name(csh handle, unsigned int id)
{
#ifndef CAPSTONE_DIET
	if (id >= MIPS_INS_ENDING)
		return NULL;

	return insn_name_maps[id].name;
#else
	return NULL;
#endif
}

#ifndef CAPSTONE_DIET
static const name_map group_name_maps[] = {
	// generic groups
	{ MIPS_GRP_INVALID, NULL },
	{ MIPS_GRP_JUMP, "jump" },
	{ MIPS_GRP_CALL, "call" },
	{ MIPS_GRP_RET, "ret" },
	{ MIPS_GRP_INT, "int" },
	{ MIPS_GRP_IRET, "iret" },
	{ MIPS_GRP_PRIVILEGE, "privileged" },
	{ MIPS_GRP_BRANCH_RELATIVE, "branch_relative" },

	// architecture-specific groups
	{ MIPS_GRP_BITCOUNT, "bitcount" },
	{ MIPS_GRP_DSP, "dsp" },
	{ MIPS_GRP_DSPR2, "dspr2" },
	{ MIPS_GRP_FPIDX, "fpidx" },
	{ MIPS_GRP_MSA, "msa" },
	{ MIPS_GRP_MIPS32R2, "mips32r2" },
	{ MIPS_GRP_MIPS64, "mips64" },
	{ MIPS_GRP_MIPS64R2, "mips64r2" },
	{ MIPS_GRP_SEINREG, "seinreg" },
	{ MIPS_GRP_STDENC, "stdenc" },
	{ MIPS_GRP_SWAP, "swap" },
	{ MIPS_GRP_MICROMIPS, "micromips" },
	{ MIPS_GRP_MIPS16MODE, "mips16mode" },
	{ MIPS_GRP_FP64BIT, "fp64bit" },
	{ MIPS_GRP_NONANSFPMATH, "nonansfpmath" },
	{ MIPS_GRP_NOTFP64BIT, "notfp64bit" },
	{ MIPS_GRP_NOTINMICROMIPS, "notinmicromips" },
	{ MIPS_GRP_NOTNACL, "notnacl" },

	{ MIPS_GRP_NOTMIPS32R6, "notmips32r6" },
	{ MIPS_GRP_NOTMIPS64R6, "notmips64r6" },
	{ MIPS_GRP_CNMIPS, "cnmips" },

	{ MIPS_GRP_MIPS32, "mips32" },
	{ MIPS_GRP_MIPS32R6, "mips32r6" },
	{ MIPS_GRP_MIPS64R6, "mips64r6" },

	{ MIPS_GRP_MIPS2, "mips2" },
	{ MIPS_GRP_MIPS3, "mips3" },
	{ MIPS_GRP_MIPS3_32, "mips3_32"},
	{ MIPS_GRP_MIPS3_32R2, "mips3_32r2" },

	{ MIPS_GRP_MIPS4_32, "mips4_32" },
	{ MIPS_GRP_MIPS4_32R2, "mips4_32r2" },
	{ MIPS_GRP_MIPS5_32R2, "mips5_32r2" },

	{ MIPS_GRP_GP32BIT, "gp32bit" },
	{ MIPS_GRP_GP64BIT, "gp64bit" },
};
#endif

const char *Mips_group_name(csh handle, unsigned int id)
{
#ifndef CAPSTONE_DIET
	return id2name(group_name_maps, ARR_SIZE(group_name_maps), id);
#else
	return NULL;
#endif
}

// map instruction name to public instruction ID
mips_reg Mips_map_insn(const char *name)
{
	// handle special alias first
	unsigned int i;

	// NOTE: skip first NULL name in insn_name_maps
	i = name2id(&insn_name_maps[1], ARR_SIZE(insn_name_maps) - 1, name);

	return (i != -1)? i : MIPS_REG_INVALID;
}

// map internal raw register to 'public' register
mips_reg Mips_map_register(unsigned int r)
{
	// for some reasons different Mips modes can map different register number to
	// the same Mips register. this function handles the issue for exposing Mips
	// operands by mapping internal registers to 'public' register.
	static const unsigned int map[] = { 0,
		MIPS_REG_AT, MIPS_REG_DSPCCOND, MIPS_REG_DSPCARRY, MIPS_REG_DSPEFI, MIPS_REG_DSPOUTFLAG,
		MIPS_REG_DSPPOS, MIPS_REG_DSPSCOUNT, MIPS_REG_FP, MIPS_REG_GP, MIPS_REG_2,
		MIPS_REG_1, MIPS_REG_0, MIPS_REG_6, MIPS_REG_4, MIPS_REG_5,
		MIPS_REG_3, MIPS_REG_7, MIPS_REG_PC, MIPS_REG_RA, MIPS_REG_SP,
		MIPS_REG_ZERO, MIPS_REG_A0, MIPS_REG_A1, MIPS_REG_A2, MIPS_REG_A3,
		MIPS_REG_AC0, MIPS_REG_AC1, MIPS_REG_AC2, MIPS_REG_AC3, MIPS_REG_AT,
		MIPS_REG_CC0, MIPS_REG_CC1, MIPS_REG_CC2, MIPS_REG_CC3, MIPS_REG_CC4,
		MIPS_REG_CC5, MIPS_REG_CC6, MIPS_REG_CC7, MIPS_REG_0, MIPS_REG_1,
		MIPS_REG_2, MIPS_REG_3, MIPS_REG_4, MIPS_REG_5, MIPS_REG_6,
		MIPS_REG_7, MIPS_REG_8, MIPS_REG_9, MIPS_REG_0, MIPS_REG_1,
		MIPS_REG_2, MIPS_REG_3, MIPS_REG_4, MIPS_REG_5, MIPS_REG_6,
		MIPS_REG_7, MIPS_REG_8, MIPS_REG_9, MIPS_REG_10, MIPS_REG_11,
		MIPS_REG_12, MIPS_REG_13, MIPS_REG_14, MIPS_REG_15, MIPS_REG_16,
		MIPS_REG_17, MIPS_REG_18, MIPS_REG_19, MIPS_REG_20, MIPS_REG_21,
		MIPS_REG_22, MIPS_REG_23, MIPS_REG_24, MIPS_REG_25, MIPS_REG_26,
		MIPS_REG_27, MIPS_REG_28, MIPS_REG_29, MIPS_REG_30, MIPS_REG_31,
		MIPS_REG_10, MIPS_REG_11, MIPS_REG_12, MIPS_REG_13, MIPS_REG_14,
		MIPS_REG_15, MIPS_REG_16, MIPS_REG_17, MIPS_REG_18, MIPS_REG_19,
		MIPS_REG_20, MIPS_REG_21, MIPS_REG_22, MIPS_REG_23, MIPS_REG_24,
		MIPS_REG_25, MIPS_REG_26, MIPS_REG_27, MIPS_REG_28, MIPS_REG_29,
		MIPS_REG_30, MIPS_REG_31, MIPS_REG_F0, MIPS_REG_F2, MIPS_REG_F4,
		MIPS_REG_F6, MIPS_REG_F8, MIPS_REG_F10, MIPS_REG_F12, MIPS_REG_F14,
		MIPS_REG_F16, MIPS_REG_F18, MIPS_REG_F20, MIPS_REG_F22, MIPS_REG_F24,
		MIPS_REG_F26, MIPS_REG_F28, MIPS_REG_F30, MIPS_REG_DSPOUTFLAG20, MIPS_REG_DSPOUTFLAG21,
		MIPS_REG_DSPOUTFLAG22, MIPS_REG_DSPOUTFLAG23, MIPS_REG_F0, MIPS_REG_F1, MIPS_REG_F2,
		MIPS_REG_F3, MIPS_REG_F4, MIPS_REG_F5, MIPS_REG_F6, MIPS_REG_F7,
		MIPS_REG_F8, MIPS_REG_F9, MIPS_REG_F10, MIPS_REG_F11, MIPS_REG_F12,
		MIPS_REG_F13, MIPS_REG_F14, MIPS_REG_F15, MIPS_REG_F16, MIPS_REG_F17,
		MIPS_REG_F18, MIPS_REG_F19, MIPS_REG_F20, MIPS_REG_F21, MIPS_REG_F22,
		MIPS_REG_F23, MIPS_REG_F24, MIPS_REG_F25, MIPS_REG_F26, MIPS_REG_F27,
		MIPS_REG_F28, MIPS_REG_F29, MIPS_REG_F30, MIPS_REG_F31, MIPS_REG_FCC0,
		MIPS_REG_FCC1, MIPS_REG_FCC2, MIPS_REG_FCC3, MIPS_REG_FCC4, MIPS_REG_FCC5,
		MIPS_REG_FCC6, MIPS_REG_FCC7, MIPS_REG_0, MIPS_REG_1, MIPS_REG_2,
		MIPS_REG_3, MIPS_REG_4, MIPS_REG_5, MIPS_REG_6, MIPS_REG_7,
		MIPS_REG_8, MIPS_REG_9, MIPS_REG_10, MIPS_REG_11, MIPS_REG_12,
		MIPS_REG_13, MIPS_REG_14, MIPS_REG_15, MIPS_REG_16, MIPS_REG_17,
		MIPS_REG_18, MIPS_REG_19, MIPS_REG_20, MIPS_REG_21, MIPS_REG_22,
		MIPS_REG_23, MIPS_REG_24, MIPS_REG_25, MIPS_REG_26, MIPS_REG_27,
		MIPS_REG_28, MIPS_REG_29, MIPS_REG_30, MIPS_REG_31, MIPS_REG_FP,
		MIPS_REG_F0, MIPS_REG_F1, MIPS_REG_F2, MIPS_REG_F3, MIPS_REG_F4,
		MIPS_REG_F5, MIPS_REG_F6, MIPS_REG_F7, MIPS_REG_F8, MIPS_REG_F9,
		MIPS_REG_F10, MIPS_REG_F11, MIPS_REG_F12, MIPS_REG_F13, MIPS_REG_F14,
		MIPS_REG_F15, MIPS_REG_F16, MIPS_REG_F17, MIPS_REG_F18, MIPS_REG_F19,
		MIPS_REG_F20, MIPS_REG_F21, MIPS_REG_F22, MIPS_REG_F23, MIPS_REG_F24,
		MIPS_REG_F25, MIPS_REG_F26, MIPS_REG_F27, MIPS_REG_F28, MIPS_REG_F29,
		MIPS_REG_F30, MIPS_REG_F31, MIPS_REG_GP, MIPS_REG_AC0, MIPS_REG_AC1,
		MIPS_REG_AC2, MIPS_REG_AC3, 0, 0, 0,
		0, MIPS_REG_4, MIPS_REG_5, MIPS_REG_6, MIPS_REG_7,
		MIPS_REG_8, MIPS_REG_9, MIPS_REG_10, MIPS_REG_11, MIPS_REG_12,
		MIPS_REG_13, MIPS_REG_14, MIPS_REG_15, MIPS_REG_16, MIPS_REG_17,
		MIPS_REG_18, MIPS_REG_19, MIPS_REG_20, MIPS_REG_21, MIPS_REG_22,
		MIPS_REG_23, MIPS_REG_24, MIPS_REG_25, MIPS_REG_26, MIPS_REG_27,
		MIPS_REG_28, MIPS_REG_29, MIPS_REG_30, MIPS_REG_31, MIPS_REG_K0,
		MIPS_REG_K1, MIPS_REG_AC0, MIPS_REG_AC1, MIPS_REG_AC2, MIPS_REG_AC3,
		MIPS_REG_MPL0, MIPS_REG_MPL1, MIPS_REG_MPL2, MIPS_REG_P0, MIPS_REG_P1,
		MIPS_REG_P2, MIPS_REG_RA, MIPS_REG_S0, MIPS_REG_S1, MIPS_REG_S2,
		MIPS_REG_S3, MIPS_REG_S4, MIPS_REG_S5, MIPS_REG_S6, MIPS_REG_S7,
		MIPS_REG_SP, MIPS_REG_T0, MIPS_REG_T1, MIPS_REG_T2, MIPS_REG_T3,
		MIPS_REG_T4, MIPS_REG_T5, MIPS_REG_T6, MIPS_REG_T7, MIPS_REG_T8,
		MIPS_REG_T9, MIPS_REG_V0, MIPS_REG_V1, MIPS_REG_W0, MIPS_REG_W1,
		MIPS_REG_W2, MIPS_REG_W3, MIPS_REG_W4, MIPS_REG_W5, MIPS_REG_W6,
		MIPS_REG_W7, MIPS_REG_W8, MIPS_REG_W9, MIPS_REG_W10, MIPS_REG_W11,
		MIPS_REG_W12, MIPS_REG_W13, MIPS_REG_W14, MIPS_REG_W15, MIPS_REG_W16,
		MIPS_REG_W17, MIPS_REG_W18, MIPS_REG_W19, MIPS_REG_W20, MIPS_REG_W21,
		MIPS_REG_W22, MIPS_REG_W23, MIPS_REG_W24, MIPS_REG_W25, MIPS_REG_W26,
		MIPS_REG_W27, MIPS_REG_W28, MIPS_REG_W29, MIPS_REG_W30, MIPS_REG_W31,
		MIPS_REG_ZERO, MIPS_REG_A0, MIPS_REG_A1, MIPS_REG_A2, MIPS_REG_A3,
		MIPS_REG_AC0, MIPS_REG_F0, MIPS_REG_F1, MIPS_REG_F2, MIPS_REG_F3,
		MIPS_REG_F4, MIPS_REG_F5, MIPS_REG_F6, MIPS_REG_F7, MIPS_REG_F8,
		MIPS_REG_F9, MIPS_REG_F10, MIPS_REG_F11, MIPS_REG_F12, MIPS_REG_F13,
		MIPS_REG_F14, MIPS_REG_F15, MIPS_REG_F16, MIPS_REG_F17, MIPS_REG_F18,
		MIPS_REG_F19, MIPS_REG_F20, MIPS_REG_F21, MIPS_REG_F22, MIPS_REG_F23,
		MIPS_REG_F24, MIPS_REG_F25, MIPS_REG_F26, MIPS_REG_F27, MIPS_REG_F28,
		MIPS_REG_F29, MIPS_REG_F30, MIPS_REG_F31, MIPS_REG_DSPOUTFLAG16_19, MIPS_REG_HI,
		MIPS_REG_K0, MIPS_REG_K1, MIPS_REG_LO, MIPS_REG_S0, MIPS_REG_S1,
		MIPS_REG_S2, MIPS_REG_S3, MIPS_REG_S4, MIPS_REG_S5, MIPS_REG_S6,
		MIPS_REG_S7, MIPS_REG_T0, MIPS_REG_T1, MIPS_REG_T2, MIPS_REG_T3,
		MIPS_REG_T4, MIPS_REG_T5, MIPS_REG_T6, MIPS_REG_T7, MIPS_REG_T8,
		MIPS_REG_T9, MIPS_REG_V0, MIPS_REG_V1
	};

	if (r < ARR_SIZE(map))
		return map[r];

	// cannot find this register
	return 0;
}

#endif
