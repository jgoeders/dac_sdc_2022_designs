// #define DEBUG

#ifdef DEBUG

#endif
#include <fstream>
#include <iostream>
using namespace std;
#include "config.h"
#include "conv1x1DSP2.hpp"
#include "conv2d_DSPopt.hpp"
#include "conv2d_l0.hpp"
#include "function.h"
#include "pool_reord.hpp"
#include "stream_tools.h"
#include "weights.hpp"
#include "Resize_opr_linear_opt.h"
#include <ap_int.h>
#include <hls_video.h>
#include <iostream>
#include <stdint.h>

#define IN_IMAGE_WIDTH 640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160



void resize_nearest_2pack(hls::stream<ap_uint<24 * 2> > &in, hls::stream<ap_uint<24> > &out)
{
    static const ap_uint<1> row_mask[IN_IMAGE_HEIGHT] = {
        1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 
        0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
        1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 
        0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
        1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 
    };

    #pragma HLS DATAFLOW

    for (unsigned r = 0; r < IN_IMAGE_HEIGHT; ++r)
    {
        for (unsigned c = 0; c < IN_IMAGE_WIDTH; c += 2)
        {
          #pragma HLS PIPELINE II=1
            ap_uint<24> in_data0, in_data1;
            (in_data1, in_data0) = in.read();
            if (row_mask[r] == 0x1)
            {
                ap_uint<24> out_data = in_data0;
                out.write(out_data);
            }
        }
    }
};






void stream_to_mat(
    hls::stream<ap_uint<24> > &in,
    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> &raw_img) {

  for (int i = 0; i < IN_IMAGE_HEIGHT; i++) {
    for (int j = 0; j < IN_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
      hls::Scalar<3, ap_uint<8> > pix;
      ap_uint<24> in_data = in.read();
      for (unsigned int p = 0; p < 3; p++) {

        pix.val[p] = in_data(8 * p + 7, 8 * p);
      }
      raw_img << pix;
    }
  }
}

void mat_to_stream(
    hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> &resize_img,
    hls::stream<ap_uint<24> > &out) {

  for (int i = 0; i < RESIZE_IMAGE_HEIGHT; i++) {
    for (int j = 0; j < RESIZE_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
      hls::Scalar<3, ap_uint<8> > pix;
      resize_img >> pix;
      ap_uint<24> out_data;
      for (unsigned int p = 0; p < 3; p++) {
        out_data(8 * p + 7, 8 * p) = pix.val[p];
      }
      out.write(out_data);
    }
  }
}

void resize(hls::stream<ap_uint<24> > &in, hls::stream<ap_uint<24> > &out) {
#pragma HLS dataflow
  hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;
#pragma HLS STREAM variable = raw_img depth = 1024 dim = 1
  hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;
#pragma HLS STREAM variable = resize_img depth = 1024 dim = 1
  stream_to_mat(in, raw_img);
  // hls::Resize(raw_img, resize_img, HLS_INTER_LINEAR);
  // hls::Resize_opr_linear(raw_img, resize_img);
  Resize_opr_linear_try(raw_img, resize_img);
  mat_to_stream(resize_img, out);
}

void resize_batch(hls::stream<ap_uint<24> > &in, hls::stream<ap_uint<24> > &out,
                  unsigned int reps) {
  for (unsigned int rep = 0; rep < reps; rep++) {
    resize(in, out);
  }
}


void resize_batch_simd2(hls::stream<ap_uint<24 * 2> > &in, hls::stream<ap_uint<24> > &out,
                  unsigned int reps) {
  for (unsigned int rep = 0; rep < reps; rep++) {
    // resize_nearest_2pack(in, out);
    Resize_opr_linear_simd2(in, out);
  }
}

void do_compute2(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
                 const unsigned int reps = 1) {
#pragma HLS DATAFLOW

  const unsigned int num_per_rep = 360 * 640 * 3 * 8 / 64;

  hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable = in_stream_extract depth = 1024 dim = 1
  ExtractPixels<64, num_per_rep>(in, in_stream_extract, reps);

  hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable = in_stream0 depth = 1024 dim = 1
  StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract,
                                                             in_stream0, reps);


//   hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream1("in_stream1");
// #pragma HLS STREAM variable = in_stream1 depth = 32 dim = 1

//   StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT * CONV_0_IFM_CH,
//                                     num_per_rep / 3>(in_stream0, in_stream1,
//                                                      reps);


  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH * 2> > in_stream1("in_stream1");
#pragma HLS STREAM variable = in_stream1 depth = 32 dim = 1

  StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT * CONV_0_IFM_CH * 2,
                                    num_per_rep / 3>(in_stream0, in_stream1,
                                                     reps);
#ifdef DEBUG
  cout << "in_stream1 size " << in_stream1.size() << endl;

#endif

  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream2("in_stream2");
#pragma HLS STREAM variable = in_stream2 depth = 32 dim = 1
  resize_batch_simd2(in_stream1, in_stream2, reps);
#ifdef DEBUG
  cout << "in_stream2 size " << in_stream2.size() << endl;
#endif
  //***********CONV_0*************
  hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2> > conv_0_out(
      "conv_0_out");
#pragma HLS STREAM variable = conv_0_out depth = 128 dim = 1
  conv3x3_l0_bn_act_DSPopt<CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_IFM_CH,
                           CONV_0_IN_BIT, CONV_0_OFM_CH, CONV_0_OUT_BIT,
                           CONV_0_W_BIT, 26, CONV_0_INC_BIT_NEW,
                           CONV_0_BIAS_BIT_NEW, CONV_0_SIMD_DSP6, 3,
                           CONV_0_INPE, CONV_0_PE_DSP6, CONV_0_L_SHIFT>(
      in_stream2, conv_0_w_new, conv_0_inc_new, conv_0_bias_new, conv_0_out,
      reps);
#ifdef DEBUG
  cout << "conv_0_out size " << conv_0_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_0_OFM_ROW, CONV_0_OFM_COL,
                                   CONV_0_OFM_CH, CONV_0_PE_DSP6,
                                   CONV_0_OUT_BIT>(conv_0_out,
                                                   "conv_0_DSP6_out.txt", reps);
#endif

  //***********POOL_0*************
  hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2> > pool_0_out(
      "pool_0_out");
#pragma HLS STREAM variable = pool_0_out depth = 128 dim = 1
  max_pool2x2<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH, CONV_0_OUT_BIT,
              CONV_0_PE_DSP6>(conv_0_out, pool_0_out, reps);
#ifdef DEBUG
  cout << "pool_0_out size " << pool_0_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_1_IFM_ROW, CONV_1_IFM_COL,
                                   CONV_1_IFM_CH, CONV_1_INPE, CONV_0_OUT_BIT>(
      pool_0_out, "pool_0_DSP6_out.txt", reps);
#endif

  //***********CONV_1*************
  hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_PE_DSP6 * 2> > conv_1_out(
      "conv_1_out");
#pragma HLS STREAM variable = conv_1_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_1_IFM_ROW, CONV_1_IFM_COL, CONV_1_IFM_CH,
                        CONV_1_IN_BIT, CONV_1_OFM_CH, CONV_1_OUT_BIT,
                        CONV_1_W_BIT, 16, CONV_1_INC_BIT_NEW,
                        CONV_1_BIAS_BIT_NEW, CONV_1_SIMD_DSP6, 4, CONV_1_INPE,
                        CONV_1_PE_DSP6, CONV_1_L_SHIFT, 4>(
      pool_0_out, conv_1_w_new, conv_1_inc_new, conv_1_bias_new, conv_1_out,
      reps);
#ifdef DEBUG
  cout << "conv_1_out size " << conv_1_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_1_OFM_ROW, CONV_1_OFM_COL,
                                   CONV_1_OFM_CH, CONV_1_PE_DSP6,
                                   CONV_1_OUT_BIT>(conv_1_out,
                                                   "conv_1_DSP6_out.txt", reps);
#endif

  //***********POOL_1*************
  hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_PE_DSP6 * 2> > pool_1_out(
      "pool_1_out");
#pragma HLS STREAM variable = pool_1_out depth = 128 dim = 1
  max_pool2x2<CONV_1_OFM_ROW, CONV_1_OFM_COL, CONV_1_OFM_CH, CONV_1_OUT_BIT,
              CONV_1_PE_DSP6>(conv_1_out, pool_1_out, reps);
#ifdef DEBUG
  cout << "pool_1_out size " << pool_1_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_2_IFM_ROW, CONV_2_IFM_COL,
                                   CONV_2_IFM_CH, CONV_2_INPE, CONV_1_OUT_BIT>(
      pool_1_out, "pool_1_DSP6_out.txt", reps);
#endif

  //***********CONV_2*************
  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE_DSP6 * 2> > conv_2_out(
      "conv_2_out");
#pragma HLS STREAM variable = conv_2_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH,
                        CONV_2_IN_BIT, CONV_2_OFM_CH, CONV_2_OUT_BIT,
                        CONV_2_W_BIT, 17, CONV_2_INC_BIT_NEW,
                        CONV_2_BIAS_BIT_NEW, CONV_2_SIMD_DSP6, 4, CONV_2_INPE,
                        CONV_2_PE_DSP6, CONV_2_L_SHIFT, 2>(
      pool_1_out, conv_2_w_new, conv_2_inc_new, conv_2_bias_new, conv_2_out,
      reps);
#ifdef DEBUG
  cout << "conv_2_out size " << conv_2_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_2_OFM_ROW, CONV_2_OFM_COL,
                                   CONV_2_OFM_CH, CONV_2_PE_DSP6,
                                   CONV_2_OUT_BIT>(conv_2_out,
                                                   "conv_2_DSP6_out.txt", reps);
#endif

  //***********POOL_2*************
  hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_PE_DSP6 * 2> > pool_2_out(
      "pool_2_out");
#pragma HLS STREAM variable = pool_2_out depth = 128 dim = 1
  max_pool2x2<CONV_2_OFM_ROW, CONV_2_OFM_COL, CONV_2_OFM_CH, CONV_2_OUT_BIT,
              CONV_2_PE_DSP6>(conv_2_out, pool_2_out, reps);
#ifdef DEBUG
  cout << "pool_2_out size " << pool_2_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_3_IFM_ROW, CONV_3_IFM_COL,
                                   CONV_3_IFM_CH, CONV_3_INPE, CONV_2_OUT_BIT>(
      pool_2_out, "pool_2_DSP6_out.txt", reps);
#endif

  //***********CONV_3*************
  hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_PE_DSP6 * 2> > conv_3_out(
      "conv_3_out");
#pragma HLS STREAM variable = conv_3_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_3_IFM_ROW, CONV_3_IFM_COL, CONV_3_IFM_CH,
                        CONV_3_IN_BIT, CONV_3_OFM_CH, CONV_3_OUT_BIT,
                        CONV_3_W_BIT, 18, CONV_3_INC_BIT_NEW,
                        CONV_3_BIAS_BIT_NEW, CONV_3_SIMD_DSP6, 4, CONV_3_INPE,
                        CONV_3_PE_DSP6, CONV_3_L_SHIFT, 1>(
      pool_2_out, conv_3_w_new, conv_3_inc_new, conv_3_bias_new, conv_3_out,
      reps);
#ifdef DEBUG
  cout << "conv_3_out size " << conv_3_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_3_OFM_ROW, CONV_3_OFM_COL,
                                   CONV_3_OFM_CH, CONV_3_PE_DSP6,
                                   CONV_3_OUT_BIT>(conv_3_out,
                                                   "conv_3_DSP6_out.txt", reps);
#endif

  //***********POOL_3*************
  hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_PE_DSP6 * 2> > pool_3_out(
      "pool_3_out");
#pragma HLS STREAM variable = pool_3_out depth = 128 dim = 1
  max_pool2x2<CONV_3_OFM_ROW, CONV_3_OFM_COL, CONV_3_OFM_CH, CONV_3_OUT_BIT,
              CONV_3_PE_DSP6>(conv_3_out, pool_3_out, reps);
#ifdef DEBUG
  cout << "pool_3_out size " << pool_3_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_4_IFM_ROW, CONV_4_IFM_COL,
                                   CONV_4_IFM_CH, CONV_4_INPE, CONV_3_OUT_BIT>(
      pool_3_out, "pool_3_DSP6_out.txt", reps);
#endif

  //***********CONV_4*************
  hls::stream<ap_uint<CONV_4_OUT_BIT * CONV_4_PE_DSP6 * 2> > conv_4_out(
      "conv_4_out");
#pragma HLS STREAM variable = conv_4_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_4_IFM_ROW, CONV_4_IFM_COL, CONV_4_IFM_CH,
                        CONV_4_IN_BIT, CONV_4_OFM_CH, CONV_4_OUT_BIT,
                        CONV_4_W_BIT, 18, CONV_4_INC_BIT_NEW,
                        CONV_4_BIAS_BIT_NEW, CONV_4_SIMD_DSP6, 4, CONV_4_INPE,
                        CONV_4_PE_DSP6, CONV_4_L_SHIFT, 1>(
      pool_3_out, conv_4_w_new, conv_4_inc_new, conv_4_bias_new, conv_4_out,
      reps);
#ifdef DEBUG
  cout << "conv_4_out size " << conv_4_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_4_OFM_ROW, CONV_4_OFM_COL,
                                   CONV_4_OFM_CH, CONV_4_PE_DSP6,
                                   CONV_4_OUT_BIT>(conv_4_out,
                                                   "conv_4_DSP6_out.txt", reps);
#endif

  //***********CONV_5*************
  hls::stream<ap_uint<CONV_5_OUT_BIT * CONV_5_PE_DSP6 * 2> > conv_5_out(
      "conv_5_out");
#pragma HLS STREAM variable = conv_5_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_5_IFM_ROW, CONV_5_IFM_COL, CONV_5_IFM_CH,
                        CONV_5_IN_BIT, CONV_5_OFM_CH, CONV_5_OUT_BIT,
                        CONV_5_W_BIT, 18, CONV_5_INC_BIT_NEW,
                        CONV_5_BIAS_BIT_NEW, CONV_5_SIMD_DSP6, 4, CONV_5_INPE,
                        CONV_5_PE_DSP6, CONV_5_L_SHIFT, 1>(
      conv_4_out, conv_5_w_new, conv_5_inc_new, conv_5_bias_new, conv_5_out,
      reps);
#ifdef DEBUG
  cout << "conv_5_out size " << conv_5_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_5_OFM_ROW, CONV_5_OFM_COL,
                                   CONV_5_OFM_CH, CONV_5_PE_DSP6,
                                   CONV_5_OUT_BIT>(conv_5_out,
                                                   "conv_5_DSP6_out.txt", reps);
#endif

  //***********CONV_6*************
  hls::stream<ap_uint<CONV_6_OUT_BIT * CONV_6_PE_DSP6 * 2> > conv_6_out(
      "conv_6_out");
#pragma HLS STREAM variable = conv_6_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_6_IFM_ROW, CONV_6_IFM_COL, CONV_6_IFM_CH,
                        CONV_6_IN_BIT, CONV_6_OFM_CH, CONV_6_OUT_BIT,
                        CONV_6_W_BIT, 18, CONV_6_INC_BIT_NEW,
                        CONV_6_BIAS_BIT_NEW, CONV_6_SIMD_DSP6, 4, CONV_6_INPE,
                        CONV_6_PE_DSP6, CONV_6_L_SHIFT, 1>(
      conv_5_out, conv_6_w_new, conv_6_inc_new, conv_6_bias_new, conv_6_out,
      reps);
#ifdef DEBUG
  cout << "conv_6_out size " << conv_6_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_6_OFM_ROW, CONV_6_OFM_COL,
                                   CONV_6_OFM_CH, CONV_6_PE_DSP6,
                                   CONV_6_OUT_BIT>(conv_6_out,
                                                   "conv_6_DSP6_out.txt", reps);
#endif

  //***********CONV_7*************
  hls::stream<ap_uint<CONV_7_OUT_BIT * CONV_7_PE_DSP6 * 2> > conv_7_out(
      "conv_7_out");
#pragma HLS STREAM variable = conv_7_out depth = 128 dim = 1
  conv3x3_bn_act_DSPopt_serialACT<CONV_7_IFM_ROW, CONV_7_IFM_COL, CONV_7_IFM_CH,
                        CONV_7_IN_BIT, CONV_7_OFM_CH, CONV_7_OUT_BIT,
                        CONV_7_W_BIT, 18, CONV_7_INC_BIT_NEW,
                        CONV_7_BIAS_BIT_NEW, CONV_7_SIMD_DSP6, 4, CONV_7_INPE,
                        CONV_7_PE_DSP6, CONV_7_L_SHIFT, 1>(
      conv_6_out, conv_7_w_new, conv_7_inc_new, conv_7_bias_new, conv_7_out,
      reps);
#ifdef DEBUG
  cout << "conv_7_out size " << conv_7_out.size() << endl;
  print_mavu_DSPopt_stream_through<CONV_7_OFM_ROW, CONV_7_OFM_COL,
                                   CONV_7_OFM_CH, CONV_7_PE_DSP6,
                                   CONV_7_OUT_BIT>(conv_7_out,
                                                   "conv_7_DSP6_out.txt", reps);
#endif
  hls::stream<ap_uint<32 * CONV_8_PE_DSP2> > conv_8_out("conv_8_out");
#pragma HLS STREAM variable = conv_8_out depth = 64 dim = 1
  conv1x1_DSPopt<CONV_8_IFM_ROW, CONV_8_IFM_COL, CONV_8_IFM_CH, CONV_8_IN_BIT,
                 CONV_8_OFM_CH, CONV_8_W_BIT, CONV_8_BIAS_BIT_NEW, 32,
                 CONV_8_SIMD_DSP2, CONV_8_PE_DSP2, CONV_8_INPE>(
      conv_7_out, conv_8_w_new, conv_8_bias_new, conv_8_out, reps);

  AddLast<CONV_8_OFM_ROW * CONV_8_OFM_COL * CONV_8_OFM_CH / 2>(conv_8_out, out,
                                                               reps);
}

void ultra_net(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
               const unsigned reps) {

#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_0_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_0_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_1_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_2_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_3_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_4_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_6_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_w_new complete dim = 2
#pragma HLS ARRAY_PARTITION variable = conv_7_inc_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias_new complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w_new complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_8_bias_new complete dim = 1
  do_compute2(in, out, reps);
}

// #ifdef DEBUG

void load_data(const char *path, char *ptr, unsigned int size) {
  std::ifstream f(path, std::ios::in | std::ios::binary);
  if (!f) {
    std::cout << "no such file,please check the file name!/n";
    exit(0);
  }
  f.read(ptr, size);
  f.close();
}

void write_data(const char *path, char *ptr, unsigned int size) {
  std::ofstream f(path, std::ios::out | std::ios::binary);
  if (!f) {
    std::cout << "write no such file,please check the file name!/n";
    exit(0);
  }
  f.write(ptr, size);
  f.close();
}

