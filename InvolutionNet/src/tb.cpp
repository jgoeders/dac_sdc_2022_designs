#define DEBUG



#ifndef  __ultranet__
#define __ultranet__
#include "stream_tools.h"
void ultra_net(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps);
void load_data(const char *path, char *ptr, unsigned int size);
void write_data(const char *path, char *ptr, unsigned int size);
int test_single_layer();
#endif



// #include "ultranet.h"
#include <stdint.h>
#include <ap_int.h>
#include <hls_video.h>
#include "stream_tools.h"
#include <iostream>
#include <fstream>
using namespace std;
#define HW_COSIM 1
#define grid_row 10
#define grid_col 20
#define div 15*127
#define BS 1



int main(int argc, char const *argv[])
{
    // test_single_layer();
/////////////////////////////Overall/////////////////////////////////////
     unsigned char img[360][640][3];
     unsigned char a,b,c;
     a = 1;
     b = 2;
     c = a+b;

    load_data("./data/0_rgb.bin", (char *) img, sizeof(img));

    unsigned char * data = (unsigned char *) img;
    const int data_points_per_line = 8;        // ch * 10
    const int nums_line_pre_img = 360 * 640 * 3 / 8;

    int img_repeat = 1;

    hls::stream<my_ap_axis> input_stream("input stream");
    for (unsigned int rp = 0; rp < img_repeat; rp++)
    {
    	for (unsigned int i = 0; i < nums_line_pre_img; i++) {
    		my_ap_axis temp;
    		for (unsigned int j = 0; j < data_points_per_line; j++) {
    			temp.data( 8*(j+1)-1, 8*j ) = data[i * data_points_per_line + j];
    		}
    		input_stream.write(temp);
    	}
    }

    hls::stream<my_ap_axis> output_stream("output stream");
    ultra_net(input_stream, output_stream, img_repeat);

    cout << "output size :" << output_stream.size() << endl;
    // int res[10][20][6][6];
    // my_ap_axis temp;
    // int cnt = 0;
    // for (int n=0; n < 10; n ++)
    // {
    //     for (int i=0; i < 10; i ++) {
    //         for (int j=0; j < 36; j ++)
    //         {
    //             output_stream.read(temp);
    //             for (int k = 0; k < 2;k++)
    //             {
    //                 int idx = (j / 18) * 10 + i;
    //                 if (idx * 2 < 20)
    //                     res[n][idx * 2][((j % 18) * 2 + k) / 6][((j % 18) * 2 + k) % 6] = temp.data(k * 32 + 31,k * 32);
    //                 else
    //                     res[n][1 + (idx - 10) * 2][((j % 18) * 2 + k) / 6][((j % 18) * 2 + k) % 6] = temp.data(k * 32 + 31,k * 32);
    //             }
    //         }
    //     }
    // }
    // write_data("D:/VivadoHLSproject/iSmart/iSmart/src/data/boat6_0_res.bin", (char *) res, sizeof(res));
    // cout << "test....." << endl;
   

    



    ap_int<32> conv8_out [BS][grid_row*grid_col][6][6];
    for(unsigned int n = 0; n<BS; n++)
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
            for(unsigned int j = 0; j<6; j++)
                for(unsigned int k = 0; k<3; k++){
                    my_ap_axis output = output_stream.read();
                    ap_uint<64> out_data = output.data;
                    conv8_out[n][i][j][2*k]   = out_data(31,0);
                    conv8_out[n][i][j][2*k+1] = out_data(63,32);
                }



    ofstream f("conv8_out_test.txt");
    for(int i=0;i<(grid_row*grid_col);i++)
    {
        f << "conv8_out[" <<dec<<i << "]" << "[][] ="<< endl;
        for(int j=0;j<6;j++)
        {
            for(int k=0;k<6;k++)
            {
                long long int res = 0;
                res = conv8_out[0][i][j][k];
                f<< dec << res << ", ";
            }
            f << endl;
        }
        f <<endl;
        f << endl;
    }



    // for(int i = 0;i<6;i++)
    // {
    //     for(int j = 0;j<6;j++)
    //     {
    //         long long int res = 0;
    //         res = conv8_out[0][110][i][j];
    //         printf("%d,  ", res);
    //     }
    //     printf("\n");
    // }

    // float bias[6][6];
    // load_data("D:/VivadoHLSproject/4_BJTUcode/ultra_net_accelerator_code/data/last_bias.bin", (char *) bias, sizeof(bias));


    int conf [BS][grid_row*grid_col] = {0};
    for(unsigned int n = 0; n<BS; n++)
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
            for(unsigned int j = 0; j<6; j++){
                conf[n][i] += conv8_out[n][i][j][4];
            }


    
    unsigned int max_index[BS];

    int max[BS];
    for(unsigned int n = 0; n<BS; n++){
        max[n] = -9999999;
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
        {
            // printf("conf[%d] = %d\n",i,conf[n][i]);
            if(conf[n][i] > max[n]){
                max[n] = conf[n][i];
                max_index[n] = i;
            }
        }
            // if(conf[n][i] > max[n]){
            //     max[n] = conf[n][i];
            //     max_index[n] = i;
            // }
        // cout << "max index" << n  << ": " << max_index[n] << endl;
    }
    

    unsigned int grid_x[BS];
    unsigned int grid_y[BS];
    for(unsigned int n = 0; n<BS; n++){
        grid_x[n] = max_index[n] % grid_col;
        grid_y[n] = max_index[n] / grid_col;
    }

    float boxs[BS][6][4];
    for(unsigned int n = 0; n<BS; n++)
        for(unsigned int i = 0; i<6; i++)
            for(unsigned int j = 0; j<4; j++){
                // boxs[n][i][j] = conv8_out[n][max_index[n]][i][j] / float((div)) + bias[i][j];
                boxs[n][i][j] = conv8_out[n][max_index[n]][i][j] / float((div));
            }

    float x[BS] = {0}, y[BS] = {0}, w[BS] = {0}, h[BS] = {0};
    for(unsigned int n = 0; n<BS; n++){
        for(unsigned int i = 0; i<6; i++){
            x[n] += 1 / (1 + std::exp(-boxs[n][i][0]));
            y[n] += 1 / (1 + std::exp(-boxs[n][i][1]));
            w[n] += std::exp(boxs[n][i][2]);
            h[n] += std::exp(boxs[n][i][3]);
        }
        x[n] = x[n] / 6;
        y[n] = y[n] / 6;
        w[n] = w[n] / 6;
        h[n] = h[n] / 6;

        x[n] = (x[n] + grid_x[n]) * 16;
        y[n] = (y[n] + grid_y[n]) * 16;
        w[n] = w[n]*20;
        h[n] = h[n]*20;

        float xmin,xmax,ymin,ymax;

        xmin = (x[0] - w[0]/2)*640/320;
        xmax = (x[0] + w[0]/2)*640/320;
        ymin = (y[0] - h[0]/2)*360/160;
        ymax = (y[0] + h[0]/2)*360/160;

        cout << "result" << n <<" :" << endl;
        cout << "x: " << x[n] << " y: " << y[n] << " w: " << w[n] << " h: " << h[n] << endl;
        cout << "xmin: " << xmin << " xmax: " << xmax << " ymin: " << ymin << " ymax: " << ymax << endl;
    }








    return 0;
}


