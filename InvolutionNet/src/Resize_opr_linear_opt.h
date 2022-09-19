// #include "hls/hls_axi_io.h"
#include "hls_math.h"
#include "hls_stream.h"
#include <ap_int.h>
#include <hls_video.h>



#define SRC_T HLS_8UC3
// template<int ROWS,int COLS,int DROWS,int DCOLS>
void Resize_opr_linear_try (
		Mat<360, 640, SRC_T>	    &_src,
		Mat<160, 320, SRC_T>	&_dst )
{
    const int ROWS = 360;
    const int COLS = 640;
    const int DROWS = 160;
    const int DCOLS = 320;
    LineBuffer<2,(COLS>DCOLS?COLS:DCOLS)+1,Scalar<HLS_MAT_CN(SRC_T),HLS_TNAME(SRC_T)> >	    k_buf;
    Window<2,2,Scalar<HLS_MAT_CN(SRC_T),HLS_TNAME(SRC_T)> >   win;
    // short     dcols=_dst.cols;
    // short     drows=_dst.rows;
    // short     srows=_src.rows;
    // short     scols=_src.cols;
    short     dcols=DCOLS;
    short     drows=DROWS;
    short     srows=ROWS;
    short     scols=COLS;
    ap_fixed<32,16,AP_RND>    	row_rate=((ap_fixed<32,16,AP_RND> )srows)/(ap_fixed<32,16,AP_RND>)drows;
    ap_fixed<32,16,AP_RND>    	col_rate=((ap_fixed<32,16,AP_RND> )scols)/(ap_fixed<32,16,AP_RND>)dcols;
    typename filter2d_traits<HLS_TNAME(SRC_T), ap_fixed<20,2,AP_RND> ,4>::FILTER_CAST_T	u,v, u1,v1;

    Scalar<HLS_MAT_CN(SRC_T),HLS_TNAME(SRC_T)> s, temp, d;
    ap_fixed<4,2,AP_RND> 		par=0.5;
    ap_fixed<20,10,AP_RND> 		offset_row=row_rate/2-par;
    ap_fixed<20,10,AP_RND> 		offset_col=col_rate/2-par;
    ap_fixed<32,16,AP_RND> 		fx=0;
    ap_fixed<32,16,AP_RND> 		fy=0;

    short     rows=srows > drows ? srows : (drows+1);
    short     cols=scols > dcols ? scols : (dcols+1);
    assert(rows<=ROWS || rows<=DROWS+1);
    assert(cols<=COLS || cols<=DCOLS+1);

    short x=0;
    short pre_fy=-10;
    short pre_fx=-10;
    bool row_rd=false;
    bool row_wr=false;

    for(short i= 0;i<rows;i++) {
        for(short j= 0;j<cols;j++) {
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS DEPENDENCE array inter false
#pragma HLS PIPELINE

            bool col_rd=false;
            bool col_wr=false;

            short dy = row_rate>1 ? (short)(i/row_rate):(i-1);
            short dx = col_rate>1 ? (short)(j/col_rate):(j-1);

            fy =(dy)*row_rate+offset_row;
            fx =(dx)*col_rate+offset_col;

            short sx=(short)fx;
            short sy=(short)fy;
            if(fx-sx>0)
                u=fx-sx;
            else
                u=0;
            if(fy-sy>0)
                v=fy-sy;
            else
                v=0;
            u1=1-u;
            v1=1-v;
            if(sx>scols-1)
            {
                sx=scols-1;
                u=0;
            }
            if(sy>srows-1)
            {
                sy=srows-1;
                v=0;
            }
            if(j==0)
            {
                x=0;
                pre_fx=-10;
                if(row_rate>1)
                {
                    row_rd=true;
                    row_wr= (sy==(i-1)) ? true : false;
                }
                else
                {
                    if(i==0){
                        row_rd=true;
                    }
                    else if(sy!=pre_fy)
                    {
                        row_rd=true;
                        pre_fy=sy;
                    }
                    else {
                        row_rd=false;
                    }
                    row_wr= i>0? true: false;
                }
            }
            if(col_rate>1)
            {
                col_rd=true;
                col_wr= (sx==(j-1)) ? true : false;
            }else{
                if(j==0){
                    col_rd=true;
                }
                else if(sx!=pre_fx)
                {
                    col_rd=true;
                    pre_fx=sx;
                }
                else
                    col_rd=false;
                col_wr= j>0? true: false;
            }
            if(col_rd)
            {
                for(int r= 0;r<2;r++)
                {
                    win.val[r][1]=win.val[r][0];
                }
                if(row_rd)
                {

                    k_buf.val[1][x]=k_buf.val[0][x];
                    win.val[1][0]=k_buf.val[0][x];
                    if(sy<srows-1&&sx<scols-1)
                    {
                        _src >> s;
                        k_buf.val[0][x]=s;
                        win.val[0][0]=k_buf.val[0][x];
                    }
                    else if(sx>=scols-1&&sy<srows-1){
                        k_buf.val[0][x]=s;//border
                    }
                    else if(sy>=srows-1){
                        win.val[0][0]=k_buf.val[0][x];
                    }
                }
                else
                {
                    for(int r= 0;r<2;r++)
                    {
                        win.val[r][0]=k_buf.val[r][x];
                    }
                }

                x++;
            }
            if(row_wr && col_wr)
            {
                for(int k=0;k<HLS_MAT_CN(SRC_T);k++)
                {
                    typename filter2d_traits<HLS_TNAME(SRC_T), ap_fixed<15,1,AP_RND> ,4>::ACCUM_T t=0;
                    typedef typename fixed_type<HLS_TNAME(SRC_T)>::T SRCT;

                    t = ((SRCT)win.val[1][1].val[k])*u1*v1+
                        ((SRCT)win.val[1][0].val[k])*v1*u+
                        ((SRCT)win.val[0][1].val[k])*u1*v+
                        ((SRCT)win.val[0][0].val[k])*u*v;
                    d.val[k]=sr_cast<HLS_TNAME(SRC_T)>(t);
                }
                _dst << d;
            }
        }
    }
}


#define SRC_T HLS_8UC3
//C, H, W = 3, 360, 640
void Resize_opr_linear_simd2 (
        hls::stream<ap_uint<24 * 2> >      &_src,
        hls::stream<ap_uint<24> >          &_dst )
{
    // int SRC_T = HLS_8UC3;
    const int ROWS = 360;
    const int COLS = 640;
    const int DROWS = 160;
    const int DCOLS = 320;

    ap_uint<24 * 2> k_buf[2][COLS/2];
    #pragma HLS ARRAY_PARTITION variable = k_buf complete dim = 1

    ap_uint<24> win[2][2];
    #pragma HLS ARRAY_PARTITION variable = win complete dim = 1
    #pragma HLS ARRAY_PARTITION variable = win complete dim = 2

    short     dcols=DCOLS;
    short     drows=DROWS;
    short     srows=ROWS;
    short     scols=COLS;
    ap_fixed<32,16,AP_RND>      row_rate=((ap_fixed<32,16,AP_RND> )srows)/(ap_fixed<32,16,AP_RND>)drows;
    ap_fixed<32,16,AP_RND>      col_rate=((ap_fixed<32,16,AP_RND> )scols)/(ap_fixed<32,16,AP_RND>)dcols;
    typename filter2d_traits<HLS_TNAME(SRC_T), ap_fixed<20,2,AP_RND> ,4>::FILTER_CAST_T u,v, u1,v1;

    Scalar<HLS_MAT_CN(SRC_T),HLS_TNAME(SRC_T)> d;

    ap_fixed<4,2,AP_RND>        par=0.5;
    ap_fixed<20,10,AP_RND>      offset_row=row_rate/2-par;
    ap_fixed<20,10,AP_RND>      offset_col=col_rate/2-par;
    ap_fixed<32,16,AP_RND>      fx=0;
    ap_fixed<32,16,AP_RND>      fy=0;

    short     rows= srows;
    short     cols= scols;

    short x=0;
    bool row_wr=false;

    for(short i= 0;i<rows;i++) {
        for(short j= 1;j<cols;j += 2) {
#pragma HLS DEPENDENCE array inter false
#pragma HLS dependence intra false variable = k_buf
#pragma HLS dependence intra false variable = win
#pragma HLS PIPELINE II=1

            k_buf[1][x] = k_buf[0][x];
            win[1][0] = k_buf[0][x](47, 24);
            win[1][1] = k_buf[0][x](23, 0);

            bool col_wr=false;

            short dy = (short)(i/row_rate);
            short dx = (short)(j/col_rate);

            fy =(dy)*row_rate+offset_row;
            fx =(dx)*col_rate+offset_col;

            short sx=(short)fx;
            short sy=(short)fy;
            if(fx-sx>0)
                u=fx-sx;
            else
                u=0;
            if(fy-sy>0)
                v=fy-sy;
            else
                v=0;
            u1=1-u;
            v1=1-v;
            if(sx>scols-1)
            {
                sx=scols-1;
                u=0;
            }
            if(sy>srows-1)
            {
                sy=srows-1;
                v=0;
            }
            if(j==1)
            {
                x=0;
                row_wr= (sy==(i-1)) ? true : false;
            }

            if(sy<srows-1&&sx<scols-1)
            {
                k_buf[0][x] = _src.read();
                win[0][0] = k_buf[0][x](47, 24);
                win[0][1] = k_buf[0][x](23, 0);
            }
            else if(sy>=srows-1){
                win[0][0]=k_buf[0][x](47, 24);
                win[0][1]=k_buf[0][x](23, 0);
            }
            x ++;

            if(row_wr)
            {
                for(int k=0;k<HLS_MAT_CN(SRC_T);k++)
                {
                    typename filter2d_traits<HLS_TNAME(SRC_T), ap_fixed<15,1,AP_RND> ,4>::ACCUM_T t=0;
                    typedef typename fixed_type<HLS_TNAME(SRC_T)>::T SRCT;

                    t = ((SRCT)win[1][1](8*k+7, 8*k))*u1*v1+
                        ((SRCT)win[1][0](8*k+7, 8*k))*v1*u+
                        ((SRCT)win[0][1](8*k+7, 8*k))*u1*v+
                        ((SRCT)win[0][0](8*k+7, 8*k))*u*v;
                    d.val[k]=sr_cast<HLS_TNAME(SRC_T)>(t);
                }
                ap_uint<24> out_data;
                for (unsigned int p = 0; p < 3; p++) 
                {
                    out_data(8 * p + 7, 8 * p) = d.val[p];
                }
                _dst.write(out_data);
            }
        }
    }
}