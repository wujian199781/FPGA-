//*************************************************************************
// Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************

#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

#include "sliding-window-unit.h"
#include "matrix-vector-unit.h"
#include "misc.h"

// #define CONV2_DEBUG

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_ACT_KP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<K*Cin*Wbit> weights[MVTU_OutP][K*Cout/MVTU_OutP],
	const ap_int<Mbit> factorA[MVTU_OutP][Cout/MVTU_OutP], 
	const ap_int<Mbit> factorB[MVTU_OutP][Cout/MVTU_OutP],
	stream<ap_uint<Cout*Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<K*Cin*Ibit> > swu_out("swu_out");
	SWU_KP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, Cin*K, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out, weights, factorA, factorB, out_raw, reps);

	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>
	(out_raw, out, reps);

}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_ACT_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((Cin*K*K)/MVTU_InP)*(Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][Cout/MVTU_OutP], 
	const ap_int<Mbit> factorB[MVTU_OutP][Cout/MVTU_OutP], 
	stream<ap_uint<Cout*Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
	SWU_NoP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (swu_out, swu_out_reduced, reps);

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, reps);
#ifdef CONV2_DEBUG
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>
	(out_raw, out, reps);
}

template <	unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_1x1_ACT_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][(Cin/MVTU_InP)*(Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][(Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorB[MVTU_OutP][(Cout/MVTU_OutP)], 
	stream<ap_uint<Cout*Abit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned K = 1;
	const unsigned Dout = Din;

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (in, swu_out_reduced, reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
#endif

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>(out_raw, out, reps);
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP>
void CONV2D_NOACT_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((Cin*K*K)/MVTU_InP)*(Cout/MVTU_OutP)],
	stream<ap_uint<Cout*Mbit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
	SWU_NoP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (swu_out, swu_out_reduced, reps);

	stream<ap_uint<MVTU_OutP*Mbit> > out_raw("out_raw");
	MVU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP>
	(swu_out_reduced, weights, out_raw, reps);

	ExpandWidth<MVTU_OutP*Mbit, Cout*Mbit, Dout*Dout>
	(out_raw, out, reps);
}

template <	unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP>
void CONV2D_1x1_NOACT_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][(Cin/MVTU_InP)*(Cout/MVTU_OutP)], 
	stream<ap_uint<Cout*Mbit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned K = 1;
	const unsigned Dout = Din;

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (in, swu_out_reduced, reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
#endif

	stream<ap_uint<MVTU_OutP*Mbit> > out_raw("out_raw");
	MVU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP>
	(swu_out_reduced, weights, out_raw, reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Mbit, Cout*Mbit, Dout*Dout>
	(out_raw, out, reps);
}

template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_ACT_NoP_residual(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((Cin*K*K)/MVTU_InP)*(Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][Cout/MVTU_OutP], 
	const ap_int<Mbit> factorB[MVTU_OutP][Cout/MVTU_OutP], 
	stream<ap_uint<Cout*Abit> >& out, 
	stream<ap_uint<Cin*Ibit> >& out_res, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
	SWU_NoP_residual<K, S, IntermediateDout, Cin, Ibit, TopLeftPad, BottomRightPad> (samepad_out, swu_out, out_res, reps);

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (swu_out, swu_out_reduced, reps);

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, reps);

	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>
	(out_raw, out, reps);
}

template <	unsigned K,
			unsigned MAX_Din,
			unsigned MAX_Cin,
			unsigned MAX_Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_ACT_NoP_variable(
	stream<ap_uint<MAX_Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((MAX_Cin*K*K)/MVTU_InP)*(MAX_Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][(MAX_Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorB[MVTU_OutP][(MAX_Cout/MVTU_OutP)], 
	stream<ap_uint<MAX_Cout*Abit> >& out,
	const unsigned Din,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned S = 1;
	const unsigned Dout = Din;
	const unsigned IntermediateDout = (Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din) >> 1;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<MAX_Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD_variable<MAX_Cin, Ibit>(in, samepad_out, TopLeftPad, BottomRightPad, Din, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<MAX_Cin*Ibit> > swu_out("swu_out");
	SWU_NoP_variable<K, MAX_Din, MAX_Cin, Ibit> (samepad_out, swu_out, IntermediateDout, reps);

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<MAX_Cin*Ibit, MVTU_InP*Ibit, 1> (swu_out, swu_out_reduced, reps*K*K*Dout*Dout);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
#endif

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU_variable<Ibit, Wbit, Mbit, Abit, K*K*MAX_Cin, MAX_Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, Dout*Dout, /*Cin*K*K, Cout,*/ reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Abit, MAX_Cout*Abit, 1>(out_raw, out, reps*Dout*Dout);
}

template <	unsigned MAX_Din,
			unsigned MAX_Cin,
			unsigned MAX_Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_1x1_ACT_NoP_variable(
	stream<ap_uint<MAX_Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][(MAX_Cin/MVTU_InP)*(MAX_Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][(MAX_Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorB[MVTU_OutP][(MAX_Cout/MVTU_OutP)], 
	stream<ap_uint<MAX_Cout*Abit> >& out,
	const unsigned Din,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned K = 1;
	const unsigned Dout = Din;

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<MAX_Cin*Ibit, MVTU_InP*Ibit, 1> (in, swu_out_reduced, reps*K*K*Dout*Dout);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
#endif

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU_variable<Ibit, Wbit, Mbit, Abit, K*K*MAX_Cin, MAX_Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, Dout*Dout, /*Cin*K*K, Cout,*/ reps);
#ifdef CONV2_DEBUG
	cout << "swu_out_reduced.size(): " << swu_out_reduced.size() << endl;
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Abit, MAX_Cout*Abit, 1>(out_raw, out, reps*Dout*Dout);
}