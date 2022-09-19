open_project		-reset InvolutionNet_HLS
add_files 			./src/ultranet.cpp
set_top				ultra_net  
add_files -tb	    ./src/tb.cpp   


open_solution -reset "1_4ns"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 4
csynth_design


export_design -format ip_catalog
