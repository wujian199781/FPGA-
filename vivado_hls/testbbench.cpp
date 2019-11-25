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
#define TESTBENCH
// #define REAL_IMAGE
// #define DEBUG



#include "hls-nn-lib.h"
#include "halfsqueezenet-config.h"
#include "halfsqueezenet-params.h"





//#define TESTBENCH
// #define REAL_IMAGE
// #define DEBUG

//#define AP_INT_MAX_W 16384



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






/*void II_determiner(stream<ap_axis >& in, stream<ap_axis >& out) {
	halfsqueezenet(in, out, L1_Din, L1_Cin, L1_Cout, 0, 0, L2_Din, L2_Din >> 1, L2_Cin, L2_Cout, 0, 0, 1, 1);
}*/

// TESTBENCH
#ifdef TESTBENCH

#ifdef REAL_IMAGE
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main() {

const unsigned NUM_SAMPLES=1;

#ifdef REAL_IMAGE
	string imagename("test_image1.png");
	Mat im;
	im = imread(imagename.c_str(), IMREAD_COLOR);

	unsigned height = im.rows;
	unsigned width = im.cols;
#else
	unsigned height = 224;
	unsigned width = 224;
#endif

	cout << "Image height: " << height << endl;
	cout << "Image width: " << width << endl;

	const unsigned pixel_bits = L0_Ibit*L0_Cin;
	const unsigned pixels_per_line = 384/pixel_bits;
	const unsigned buffer_size = (NUM_SAMPLES*height*width)/pixels_per_line;
	stream<ap_axis > inputStream("inputStream");

	cout << "pixels_per_line: " << pixels_per_line << endl;
	cout << "buffer_size: " << buffer_size << endl;

#ifdef REAL_IMAGE
	uint8_t* pixel_ptr = (uint8_t*)im.data;
	unsigned channels = im.channels();
#else
	uint8_t* pixel_ptr = (uint8_t*)malloc(3*height*width);
	unsigned channels = 3;
	unsigned k = 0;
	for (unsigned y = 0; y < height; y++) {
		for (unsigned x = 0; x < width; x++) {
			for (unsigned c = 0; c < channels; c++) {
				pixel_ptr[y*width*channels + x*channels + c] = (k++)%256;
				// pixel_ptr[y*width*channels + x*channels + c] = 0;
			}
		}
	}
#endif
	unsigned index = 0;
	unsigned word;

	for (unsigned i = 0; i < NUM_SAMPLES; i++) {
		word = 0;
		ap_axis temp;
		for (unsigned y = 0; y < height; y++) {
			for (unsigned x = 0; x < width; x++) {
				unsigned red   = (unsigned)pixel_ptr[y*width*channels + x*channels];
				unsigned green = (unsigned)pixel_ptr[y*width*channels + x*channels + 1];
				unsigned blue  = (unsigned)pixel_ptr[y*width*channels + x*channels + 2];
				unsigned rgb   = (blue << 16) + (green << 8) + red;

				temp.data(pixel_bits*(word+1)-1, pixel_bits*word) = rgb;

				if (word == pixels_per_line-1) {
					inputStream.write(temp);
					word = 0;
					temp.data = 0;
					index++;
				}
				else
					word++;
			}
		}
	}

#ifndef REAL_IMAGE
	free(pixel_ptr);
#endif

	cout << "index: " << index << endl;
	cout << "word: " << word << endl;

	const unsigned weight_mem_size = ((L_MAX_squeeze_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP)
										+ ((L_MAX_expand_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L_MAX_expand_Cout/expand_L_MVTU_OutP);
	const unsigned factor_mem_size = L_MAX_squeeze_Cout/squeeze_L_MVTU_OutP + 2*(L_MAX_expand_Cout/expand_L_MVTU_OutP);

	stream<ap_axis> weightsfactors_stream[LAST_LAYER];

	unsigned squeeze_weight_iterations[LAST_LAYER];
	unsigned squeeze_factor_iterations[LAST_LAYER];
	unsigned expand_weight_iterations [LAST_LAYER];
	unsigned expand_factor_iterations [LAST_LAYER];

	squeeze_weight_iterations[0] = ((L1_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[1] = ((L3_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[2] = ((L5_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L5_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[3] = ((L7_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L7_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[4] = ((L9_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L9_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[5] = ((L11_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L11_Cout/squeeze_L_MVTU_OutP);
	squeeze_weight_iterations[6] = ((L13_Cin*L_K1*L_K1)/squeeze_L_MVTU_InP)*(L13_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[0] = (L1_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[1] = (L3_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[2] = (L5_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[3] = (L7_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[4] = (L9_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[5] = (L11_Cout/squeeze_L_MVTU_OutP);
	squeeze_factor_iterations[6] = (L13_Cout/squeeze_L_MVTU_OutP);

	expand_weight_iterations[0] = ((L2_Cin*L_K3*L_K3) /expand_L_MVTU_InP)*(L2_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[1] = ((L4_Cin*L_K3*L_K3) /expand_L_MVTU_InP)*(L4_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[2] = ((L6_Cin*L_K3*L_K3) /expand_L_MVTU_InP)*(L6_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[3] = ((L8_Cin*L_K3*L_K3) /expand_L_MVTU_InP)*(L8_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[4] = ((L10_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L10_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[5] = ((L12_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L12_Cout/expand_L_MVTU_OutP);
	expand_weight_iterations[6] = ((L14_Cin*L_K3*L_K3)/expand_L_MVTU_InP)*(L14_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[0] = (L2_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[1] = (L4_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[2] = (L6_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[3] = (L8_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[4] = (L10_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[5] = (L12_Cout/expand_L_MVTU_OutP);
	expand_factor_iterations[6] = (L14_Cout/expand_L_MVTU_OutP);

	ofstream ofs ("weights_file_fuck.txt", ofstream::out);
	cout << "Writing weights to ports" << endl;
	for (unsigned layer = 0; layer < LAST_LAYER; layer++) {
		ap_uint<squeeze_L_MVTU_InP*L_Wbit>* squeeze_weights[squeeze_L_MVTU_OutP];
		ap_uint<expand_L_MVTU_InP*L_Wbit>*  expand3x3_weights[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorA[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* squeeze_factorB[squeeze_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorA[expand_L_MVTU_OutP];
		ap_int<L_Mbit>* expand3x3_factorB[expand_L_MVTU_OutP];
		for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
			if (layer == 0) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights1[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA1[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB1[p];
			}
			else if (layer == 1) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights3[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA3[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB3[p];
			}
			else if (layer == 2) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights5[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA5[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB5[p];
			}
			else if (layer == 3) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights7[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA7[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB7[p];
			}
			else if (layer == 4) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights9[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA9[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB9[p];
			}
			else if (layer == 5) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights11[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA11[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB11[p];
			}
			else if (layer == 6) {
				squeeze_weights[p] = (ap_uint<squeeze_L_MVTU_InP*L_Wbit>*)weights13[p];
				squeeze_factorA[p] = (ap_int<L_Mbit>*)factorA13[p];
				squeeze_factorB[p] = (ap_int<L_Mbit>*)factorB13[p];
			}
		}
		for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
			if (layer == 0) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights2[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA2[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB2[p];
			}
			else if (layer == 1) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights4[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA4[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB4[p];
			}
			else if (layer == 2) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights6[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA6[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB6[p];
			}
			else if (layer == 3) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights8[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA8[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB8[p];
			}
			else if (layer == 4) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights10[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA10[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB10[p];
			}
			else if (layer == 5) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights12[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA12[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB12[p];
			}
			else if (layer == 6) {
				expand3x3_weights[p] = (ap_uint<expand_L_MVTU_InP*L_Wbit>*)weights14[p];
				expand3x3_factorA[p] = (ap_int<L_Mbit>*)factorA14[p];
				expand3x3_factorB[p] = (ap_int<L_Mbit>*)factorB14[p];
			}
		}

		cout << "Allocating ports for layer " << layer << endl;
		ofs << "layer: " << layer << endl;
		ofs << "squeeze_weight_iterations: " << SQUEEZE_WEIGHT_ITERATIONS << endl;
		ofs << "expand_weight_iterations: " << EXPAND_WEIGHT_ITERATIONS << endl;
		ofs << "squeeze_factor_iterations: " << SQUEEZE_FACTOR_ITERATIONS << endl;
		ofs << "expand_factor_iterations: " << EXPAND_FACTOR_ITERATIONS << endl;
		for (unsigned i = 0; i < squeeze_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
				temp.data( (p+1)*squeeze_L_MVTU_InP*L_Wbit-1, p*squeeze_L_MVTU_InP*L_Wbit ) = squeeze_weights[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < SQUEEZE_WEIGHT_ITERATIONS - squeeze_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < expand_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
				temp.data( (p+1)*expand_L_MVTU_InP*L_Wbit-1, p*expand_L_MVTU_InP*L_Wbit ) = expand3x3_weights[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_WEIGHT_ITERATIONS - expand_weight_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < squeeze_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < squeeze_L_MVTU_OutP; p++) {
				temp.data( 2*p*L_Mbit + L_Mbit-1, 2*p*L_Mbit ) = squeeze_factorA[p][i];
				temp.data( (2*p+1)*L_Mbit + L_Mbit-1, (2*p+1)*L_Mbit ) = squeeze_factorB[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < SQUEEZE_FACTOR_ITERATIONS - squeeze_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < expand_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			for (unsigned p = 0; p < expand_L_MVTU_OutP; p++) {
				temp.data( 2*p*L_Mbit + L_Mbit-1, 2*p*L_Mbit ) = expand3x3_factorA[p][i];
				temp.data( (2*p+1)*L_Mbit + L_Mbit-1, (2*p+1)*L_Mbit ) = expand3x3_factorB[p][i];
			}
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
		for (unsigned i = 0; i < EXPAND_FACTOR_ITERATIONS - expand_factor_iterations[layer]; i++) {
			ap_axis temp;
			temp.data = 0;
			weightsfactors_stream[layer].write(temp);
			ofs << hex << temp.data << dec << endl;
		}
	}
	ofs.close();
	cout << "Writing weights complete" << endl;
}

#endif
