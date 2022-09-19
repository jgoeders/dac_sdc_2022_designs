// 1. Replace the following yolo with original yolo function in ipynb
// def yolo(out_buffer, batch_n,_,result):
//     out_buffer_dataptr=out_buffer.ctypes.data_as(ctypes.c_char_p)
//     rst=np.empty( (batch_n,4),dtype=np.int32)
//     rst_dataptr=rst.ctypes.data_as(ctypes.c_char_p)
//     cfuns.yolo(out_buffer_dataptr,batch_n,rst_dataptr)
//     result.extend(rst.tolist())
// 2. when compiling image.so add -fpermissive flag

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <malloc.h>
#include <stdint.h>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdint>
#include <cmath>
#include <signal.h> 

using namespace std;

#define NUM_THREAD 16
#define ROW_SO 360
#define COL_SO 640
#define CH_SO 3


#define FIRSTLASTBITWIDTH 8
#define DIV (127 * 15)
#define GRIDROW 10
#define GRIDCOL 20
#define IMAGE_RAW_ROW  360
#define IMAGE_RAW_COL  640
#define IMAGE_ROW  160
#define IMAGE_COL  320
#define YOLO_THREAD_NUM 4



constexpr float xy_xscale=float(16*IMAGE_RAW_COL)/IMAGE_COL;
constexpr float xy_yscale=float(16*IMAGE_RAW_ROW)/IMAGE_ROW;
constexpr float wh_xscale=float(20*IMAGE_RAW_COL)/(6*IMAGE_COL);
constexpr float wh_yscale=float(20*IMAGE_RAW_ROW)/(6*IMAGE_ROW);
constexpr int vecoffset=GRIDCOL*GRIDROW*36;

int32_t* global_vec;
int32_t* global_rst;
float yolo_step;
int yolo_batch_size;

inline float sigmoid(float x){
    return 1/(1 +expf(-x) );  
}
    

void *yolo_thread(void* thr)
{
    long i=(long) thr;

    int start = (int) (yolo_step * i + 0.01);
    int end;
    if (i < YOLO_THREAD_NUM - 1) {
        end = (int)(yolo_step * (i + 1) + 0.01);
    } else {
        end = yolo_batch_size;
    }


    int32_t* vec=global_vec+start*vecoffset;
    int32_t* rst=global_rst+start*4;

    for(int bc=start;bc<end;bc++)
    {
        int32_t (*castarr)[6][6]= (int32_t (*)[6][6] ) vec;
        int max_idx=0;
        int32_t max_sum=INT32_MIN;

        for(int i=0;i<GRIDROW*GRIDCOL;i++)
        {
            int32_t sum=castarr[i][0][4]+castarr[i][1][4]+castarr[i][2][4]+castarr[i][3][4]+castarr[i][4][4]+castarr[i][5][4];
       
            if(sum>max_sum)
            {
                max_sum=sum;
                max_idx=i;
            }
        }

  

        float xy[2]={0,0};
        float wh[2]={0,0};

        
        for(int i=0;i<6;i++)
        {
            xy[0]+=sigmoid( float( castarr[max_idx][i][0])/DIV);
            xy[1]+=sigmoid( float( castarr[max_idx][i][1])/DIV);
            wh[0]+=expf(     float( castarr[max_idx][i][2])/DIV);
            wh[1]+=expf(     float( castarr[max_idx][i][3])/DIV);
        }



  

        xy[0] = (max_idx%GRIDCOL+xy[0]/6)*xy_xscale;
        xy[1] = (max_idx/GRIDCOL+xy[1]/6)*xy_yscale;




       
        wh[0] *= wh_xscale;
        wh[1] *= wh_yscale;

        float xmin = xy[0] - wh[0] / 2;
        float xmax = xy[0] + wh[0] / 2;
        float ymin = xy[1] - wh[1] / 2;
        float ymax = xy[1] + wh[1] / 2;

        rst[0]=xmin;
        rst[1]=xmax;
        rst[2]=ymin;
        rst[3]=ymax;
        rst+=4;


        vec+= vecoffset;
    }
    return thr;
}



pthread_t yolo_pthread[YOLO_THREAD_NUM];

void yolo_cpp(int32_t *vec, int batch_n, int32_t* rst){
    
    global_rst=rst;
    global_vec=vec;


    yolo_step = batch_n * 1.0 / YOLO_THREAD_NUM;
    yolo_batch_size = batch_n;
    
    for (long i=0; i<YOLO_THREAD_NUM; i++)
    {
        pthread_create(&yolo_pthread[i], NULL, yolo_thread, (void *) i); 
    }

    for (int i=0; i<YOLO_THREAD_NUM; i++)
    {
        pthread_join(yolo_pthread[i], NULL);
    }
}

extern "C" { 
    void yolo(int32_t *vec, int batch_n, int32_t* rst){
        yolo_cpp(vec, batch_n, rst);
    }  
}


