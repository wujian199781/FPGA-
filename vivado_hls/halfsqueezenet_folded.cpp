
//#define TESTBENCH
// #define REAL_IMAGE
// #define DEBUG

//#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "halfsqueezenet-config.h"
#include "halfsqueezenet-params.h"

#define L_K1 1  			   //squeeze����˴�С
#define L_K3 3  	           //expand����˴�С
#define L_S  1   		       //����
#define L_MAX_Din 58	       //ͼƬ�Ŀ�/��
#define L_MAX_squeeze_Cin  96  //squeeze����ͨ��
#define L_MAX_squeeze_Cout 32  //squeeze���ͨ��
#define L_MAX_expand_Cin   32  //expand����ͨ��
#define L_MAX_expand_Cout  96  //expand���ͨ��
#define L_Ibit L1_Ibit		   //������߳ػ�֮ǰ��λ��
#define L_Wbit L1_Wbit		   //weights��λ��
#define L_Mbit L1_Mbit	     			//factor��λ��
#define L_Abit L1_Abit 		 			//�����λ��

#define squeeze_L_MVTU_InP  L1_MVTU_InP  //��INP��Ȩ�ش����һ��
#define squeeze_L_MVTU_OutP L1_MVTU_OutP//�ֳ�OutP��
#define expand_L_MVTU_InP   L2_MVTU_InP
#define expand_L_MVTU_OutP  L2_MVTU_OutP

#define USEFUL_LINE_BITS 480
#define LINES_PER_ALL_CHANNELS 1
const unsigned NumLinesPerRep = 3136; //24*3136=224*224

#define LAST_LAYER 7

#define SQUEEZE_WEIGHT_ITERATIONS ((L_MAX_squeeze_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP)
#define SQUEEZE_FACTOR_ITERATIONS L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP
#define EXPAND_WEIGHT_ITERATIONS ((L_MAX_expand_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L_MAX_expand_Cout/expand_L_MVTU_OutP)
#define EXPAND_FACTOR_ITERATIONS L_MAX_expand_Cout/expand_L_MVTU_OutP
#define TOTAL_ITERATIONS SQUEEZE_WEIGHT_ITERATIONS+SQUEEZE_FACTOR_ITERATIONS+EXPAND_WEIGHT_ITERATIONS+EXPAND_FACTOR_ITERATIONS

static ap_uint<squeeze_L_MVTU_InP*L_Wbit> squeeze_weights[squeeze_L_MVTU_OutP][SQUEEZE_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorA[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];
static ap_int<L_Mbit> squeeze_factorB[squeeze_L_MVTU_OutP][SQUEEZE_FACTOR_ITERATIONS];

static ap_uint<expand_L_MVTU_InP*L_Wbit> expand3x3_weights[expand_L_MVTU_OutP][EXPAND_WEIGHT_ITERATIONS];
static ap_int<L_Mbit> expand3x3_factorA[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS]; 
static ap_int<L_Mbit> expand3x3_factorB[expand_L_MVTU_OutP][EXPAND_FACTOR_ITERATIONS];


template <unsigned LineWidth, unsigned NumLines>
void DemuxStream3 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	stream<ap_uint<LineWidth> >& out3, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else if (whichFire == 2)
			out2.write(temp);
		else
			out3.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void DemuxStream2 (
	stream<ap_uint<LineWidth> >& in, 
	stream<ap_uint<LineWidth> >& out1, 
	stream<ap_uint<LineWidth> >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp = in.read();
		if (whichFire == LAST_LAYER)
			out2.write(temp);
		else
			out1.write(temp);
	}
}

template <unsigned NumLines>
void DemuxStream2_0 (
	stream<ap_axis >& in, 
	stream<ap_axis >& out1, 
	stream<ap_axis >& out2, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_axis temp = in.read();
		if (whichFire == 1)
			out1.write(temp);
		else
			out2.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream3 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2, 
	stream<ap_uint<LineWidth> >& in3,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else if (whichFire == 2)
			temp = in2.read();
		else
			temp = in3.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == LAST_LAYER)
			temp = in2.read();
		else
			temp = in1.read();
		out.write(temp);
	}
}

template <unsigned LineWidth, unsigned NumLines>
void MuxStream2_0 (
	stream<ap_uint<LineWidth> >& in1, 
	stream<ap_uint<LineWidth> >& in2,
	stream<ap_uint<LineWidth> >& out, 
	const unsigned whichFire, const unsigned reps = 1)
{
	for (unsigned i = 0; i < NumLines*reps; i++) {
		ap_uint<LineWidth> temp;
		if (whichFire == 1)
			temp = in1.read();
		else
			temp = in2.read();
		out.write(temp);
	}
}

void DoFire(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din,/* const unsigned squeeze_Cin, const unsigned squeeze_Cout,*/
	const unsigned expand_Din, const unsigned expand_Din_afterpool,/* const unsigned expand_Cin, const unsigned expand_Cout,*/
	const unsigned whichFire, /*const unsigned numReps,*/
	const unsigned first_numReps,
	const unsigned conv0_numReps,
	const unsigned other_numReps,
	const unsigned pool1_numReps,
	const unsigned pool2_numReps,
	const unsigned fire5_numReps,
	const unsigned main_out_numReps,
	const unsigned final_out_numReps) 
{
#pragma HLS DATAFLOW
	stream<ap_axis> to_conv0("to_conv0");
	stream<ap_axis> to_fire("to_fire");
	DemuxStream2_0<1>(in, to_conv0, to_fire, whichFire, first_numReps);

// BRANCH 1
	stream<ap_uint<384> > in_stream_extract0("DoCompute.in_stream_extract0");
	ExtractPixels<384, NumLinesPerRep> (to_conv0, in_stream_extract0, conv0_numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("DoCompute.in_stream");
	ReduceWidth<384, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract0, in_stream, conv0_numReps);

	stream<ap_uint<L0_Cout*L0_Abit> > conv1("conv1");
	CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(in_stream, weights0, factorA0, factorB0, conv1, conv0_numReps);

	stream<ap_uint<L18_Cin*L18_Ibit> > pool1("pool1");
	POOL2D_NoP<L18_K, L18_S, L18_Din, L18_Cin, L18_Ibit> (conv1, pool1, conv0_numReps); // 224/4 = 56
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > out_padded("out_padded");
	AppendZeros<L18_Cin*L18_Ibit, L_MAX_squeeze_Cin*L_Ibit, L1_Din*L1_Din> (pool1, out_padded, conv0_numReps);//���油0

// BRANCH 2
	stream<ap_uint<USEFUL_LINE_BITS> > in_stream_extract1("DoCompute.in_stream_extract1");
	ExtractPixels<USEFUL_LINE_BITS, LINES_PER_ALL_CHANNELS> (to_fire, in_stream_extract1, other_numReps);
	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > fire_in("fire_in");
	ExpandWidth<USEFUL_LINE_BITS, L_MAX_squeeze_Cin*L_Ibit, 1> (in_stream_extract1, fire_in, other_numReps);

	stream<ap_uint<L_MAX_squeeze_Cin*L_Ibit> > first_out("first_out");
	MuxStream2_0<L_MAX_squeeze_Cin*L_Ibit, 1>(out_padded, fire_in, first_out, whichFire, squeeze_Din*squeeze_Din*1/*numReps */);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out("fire_out");
	HALFFIRE_ACT_variable<	L_K1, L_S, L_MAX_Din, L_MAX_squeeze_Cin, L_MAX_squeeze_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, squeeze_L_MVTU_InP, squeeze_L_MVTU_OutP,
							L_K3, L_S, L_MAX_Din, L_MAX_expand_Cin, L_MAX_expand_Cout, L_Ibit, L_Wbit, L_Mbit, L_Abit, expand_L_MVTU_InP, expand_L_MVTU_OutP,
							SCALE_BITS, FACTOR_SCALE_BITS>
	(first_out, squeeze_weights, squeeze_factorA, squeeze_factorB, expand3x3_weights, expand3x3_factorA, expand3x3_factorB, fire_out, 
	squeeze_Din, /*squeeze_Cin, squeeze_Cout,*/ expand_Din, /*expand_Cin, expand_Cout,*/1);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out1("fire_out1");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out2("fire_out2");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire_out3("fire_out3");
	DemuxStream3<L_MAX_expand_Cout*L_Abit, 1> (fire_out, fire_out1, fire_out2, fire_out3, whichFire, expand_Din*expand_Din*1/*numReps */);


	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out1("pool_out");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out2("pool_out");
	POOL2D_NoP<L19_K, L19_S, L19_Din, L_MAX_expand_Cout, L_Ibit> (fire_out1, pool_out1, pool1_numReps);
	POOL2D_NoP<L20_K, L20_S, L20_Din, L_MAX_expand_Cout, L_Ibit> (fire_out2, pool_out2, pool2_numReps);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > pool_out("pool_out");
	MuxStream3<L_MAX_expand_Cout*L_Abit, 1> (pool_out1, pool_out2, fire_out3, pool_out, whichFire, expand_Din_afterpool*expand_Din_afterpool*1/*numReps */);

	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > main_out("main_out");
	stream<ap_uint<L_MAX_expand_Cout*L_Abit> > fire5("fire5");
	DemuxStream2<L_MAX_expand_Cout*L_Abit, 1> (pool_out, main_out, fire5, whichFire, expand_Din_afterpool*expand_Din_afterpool*1/*numReps */);


// BRANCH 1
	stream<ap_uint<512> > main_out_padded("main_out_padded");
	AppendZeros<USEFUL_LINE_BITS, 512, 1> (main_out, main_out_padded, LINES_PER_ALL_CHANNELS*expand_Din_afterpool*expand_Din_afterpool*main_out_numReps);

// BRANCH 2
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_class("fire5_class");
	stream<ap_uint<L14_Cout*L14_Abit> > fire5_obj("fire5_obj");
#pragma HLS STREAM variable=fire5_obj depth=14*14+48
	DuplicateStreams<L14_Cout*L14_Abit, L15_Din*L15_Din>(fire5, fire5_class, fire5_obj, fire5_numReps);
	stream<ap_uint<L15_Cout*L15_Abit> > conv_class("conv_class");
	CONV2D_1x1_ACT_NoP<L15_Din, L15_Cin, L15_Cout, L15_Ibit, L15_Wbit, L15_Mbit, L15_Abit, L15_MVTU_InP, L15_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
	(fire5_class, weights15, factorA15, factorB15, conv_class, fire5_numReps);

	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out("class_out");
	ConcatStreams<L14_Cout*L14_Abit, L15_Cout*L15_Abit, L15_Din*L15_Din>(fire5_obj, conv_class, class_out, fire5_numReps);
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_obj("class_out_obj");
	stream<ap_uint<(L14_Cout+L15_Cout)*L15_Abit> > class_out_box("class_out_box");
	DuplicateStreams<(L14_Cout+L15_Cout)*L15_Abit, L16_Din*L16_Din>(class_out, class_out_obj, class_out_box, fire5_numReps);
	stream<ap_uint<L16_Cout*L16_Mbit> > conv_obj("conv_obj");
	CONV2D_1x1_NOACT_NoP<L16_Din, L16_Cin, L16_Cout, L16_Ibit, L16_Wbit, L16_Mbit, L16_MVTU_InP, L16_MVTU_OutP>
	(class_out_obj, weights16, conv_obj, fire5_numReps);
	stream<ap_uint<L17_Cout*L17_Mbit> > conv_box("conv_box");
#pragma HLS STREAM variable=conv_box depth=14*14
	CONV2D_1x1_NOACT_NoP<L17_Din, L17_Cin, L17_Cout, L17_Ibit, L17_Wbit, L17_Mbit, L17_MVTU_InP, L17_MVTU_OutP>
	(class_out_box, weights17, conv_box, fire5_numReps);
	stream<ap_uint<(L17_Cout+1)*L17_Mbit> > box_prediction("box_prediction");
	//ObjDetectSelect<L17_Mbit, L17_Cout*L17_Mbit, L17_Din*L17_Din> (conv_obj, conv_box, box_prediction, fire5_numReps);
	ConcatStreams<L17_Mbit, L17_Cout*L17_Mbit, L17_Din*L17_Din>(conv_obj, conv_box, box_prediction, fire5_numReps);
	stream<ap_uint<512> > box_prediction_padded("box_prediction_padded");
	AppendZeros<(L17_Cout+1)*L17_Mbit, 512, L17_Din*L17_Din> (box_prediction, box_prediction_padded, fire5_numReps);

	stream<ap_uint<512> > final_out("final_out");
	MuxStream2<512, 1>(main_out_padded, box_prediction_padded, final_out, whichFire, final_out_numReps);

	AddLast<1> (final_out, out, final_out_numReps);
}

void writeWeightsFactors(stream<ap_axis >& in) {
#pragma HLS DATAFLOW

	stream<ap_uint<512> > squeeze_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > squeeze_factors_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_weights_stream("squeeze_weights_stream");
	stream<ap_uint<512> > expand_factors_stream("squeeze_weights_stream");

	for (unsigned i = 0; i < TOTAL_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_axis temp_in = in.read();
		if (i < SQUEEZE_WEIGHT_ITERATIONS)
			squeeze_weights_stream.write(temp_in.data);
		else if (i < SQUEEZE_WEIGHT_ITERATIONS + EXPAND_WEIGHT_ITERATIONS)
			expand_weights_stream.write(temp_in.data);
		else if (i < SQUEEZE_WEIGHT_ITERATIONS + EXPAND_WEIGHT_ITERATIONS + SQUEEZE_FACTOR_ITERATIONS)
			squeeze_factors_stream.write(temp_in.data);
		else
			expand_factors_stream.write(temp_in.data);
	}

	for (unsigned i = 0; i < SQUEEZE_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = squeeze_weights_stream.read();
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<squeeze_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*squeeze_L_MVTU_InP*L_Wbit-1, p*squeeze_L_MVTU_InP*L_Wbit );
			squeeze_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < EXPAND_WEIGHT_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = expand_weights_stream.read();
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<expand_L_MVTU_InP*L_Wbit> temp = temp_in( (p+1)*expand_L_MVTU_InP*L_Wbit-1, p*expand_L_MVTU_InP*L_Wbit );
			expand3x3_weights[p][i] = temp;
		}
	}

	for (unsigned i = 0; i < SQUEEZE_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = squeeze_factors_stream.read();
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			squeeze_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			squeeze_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}

	for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<512> temp_in = expand_factors_stream.read();
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
#pragma HLS UNROLL
			ap_uint<2*L_Mbit> temp_factorAB = temp_in( (p+1)*2*L_Mbit-1, p*2*L_Mbit );
			expand3x3_factorA[p][i] = temp_factorAB(L_Mbit-1, 0);
			expand3x3_factorB[p][i] = temp_factorAB(2*L_Mbit-1, L_Mbit);
		}
	}
}

void halfsqueezenet(stream<ap_axis >& in, stream<ap_axis >& out,
	const unsigned squeeze_Din, /*const unsigned squeeze_Cin, const unsigned squeeze_Cout,
	const unsigned squeeze_weight_iterations, const unsigned squeeze_factor_iterations,*/
	/*const unsigned expand_Din, const unsigned expand_Din_afterpool, /*const unsigned expand_Cin, const unsigned expand_Cout,
	const unsigned expand_weight_iterations, const unsigned expand_factor_iterations,*/
	const unsigned whichFire/*, const unsigned numReps*/) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=squeeze_Din bundle=control
//#pragma HLS INTERFACE s_axilite port=squeeze_Cin bundle=control
//#pragma HLS INTERFACE s_axilite port=squeeze_Cout bundle=control
//#pragma HLS INTERFACE s_axilite port=squeeze_weight_iterations bundle=control
//#pragma HLS INTERFACE s_axilite port=squeeze_factor_iterations bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_Din bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_Din_afterpool bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_Cin bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_Cout bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_weight_iterations bundle=control
//#pragma HLS INTERFACE s_axilite port=expand_factor_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=whichFire bundle=control
//#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA15 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB15 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA15 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB15 complete dim=0
#pragma HLS RESOURCE variable=weights16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA16 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB16 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA16 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB16 complete dim=0
#pragma HLS RESOURCE variable=weights17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorA17 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB17 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorA17 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB17 complete dim=0
	
#pragma HLS RESOURCE variable=squeeze_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=squeeze_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=squeeze_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=squeeze_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=squeeze_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=squeeze_factorB complete dim=0

#pragma HLS RESOURCE variable=expand3x3_weights core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=expand3x3_factorA core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=expand3x3_factorB core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=expand3x3_weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=expand3x3_factorA complete dim=0
#pragma HLS ARRAY_PARTITION variable=expand3x3_factorB complete dim=0

	const unsigned expand_Din = squeeze_Din;
	const unsigned expand_Din_afterpool = (whichFire == 1 or whichFire == 2) ? 0.5*squeeze_Din : squeeze_Din;
	const unsigned first_numReps = (whichFire == 1) ? NumLinesPerRep*1/*numReps */ : LINES_PER_ALL_CHANNELS*squeeze_Din*squeeze_Din*1/*numReps */;
	const unsigned conv0_numReps = (whichFire == 1) ? 1/*numReps */ : 0;
	const unsigned other_numReps = (whichFire != 1) ? squeeze_Din*squeeze_Din*1/*numReps */ : 0;
	const unsigned pool1_numReps = (whichFire == 1) ? 1/*numReps */ : 0;
	const unsigned pool2_numReps = (whichFire == 2) ? 1/*numReps */ : 0;
	const unsigned fire5_numReps = (whichFire == LAST_LAYER) ? 1/*numReps */ : 0;
	const unsigned main_out_numReps = (whichFire != LAST_LAYER) ? 1/*numReps */: 0;
	const unsigned final_out_numReps = (whichFire == LAST_LAYER) ? L17_Din*L17_Din/*numReps */: LINES_PER_ALL_CHANNELS*expand_Din_afterpool*expand_Din_afterpool*1/*numReps */;

	if (whichFire == 13) {
		writeWeightsFactors(in);
	}
	else {
		DoFire(in, out,
			squeeze_Din, /*squeeze_Cin, squeeze_Cout,*/
			expand_Din, expand_Din_afterpool,/* expand_Cin, expand_Cout,*/
			whichFire,/* numReps,*/
			first_numReps,
			conv0_numReps,
			other_numReps,
			pool1_numReps,
			pool2_numReps,
			fire5_numReps,
			main_out_numReps,
			final_out_numReps);
	}
}
