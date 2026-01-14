//*** test bench ***

`timescale  1ns/1ps

module testbench  ( );
  
//*** test signal definition ***
reg          t_reset,t_clk,t_load;
reg   [3:0]  t_data_in;
wire  [15:0] lamp_show;

//*** test signal generation ***

//* t_reset *
//*** t_reset ***
initial
	begin
  		#0      t_reset = 1'b1;
		#2000   t_reset = 1'b0;
   end
   
//*** t_clk ***
initial
  begin 
    t_clk = 1'b0;
  end
   
always #50000 t_clk = ~t_clk;
  
//*** t_ena ***
initial
  begin
    #0    t_load = 1'b0;
    #2000 t_load = 1'b1;
    #3000 t_load = 1'b0;
  end

//* t_data_in *
initial
  begin
    #0    t_data_in = 4'b0000;
    #3000 t_data_in = 4'b1100;
  end  

  
//*** connect with the circuit to be tested ***
//*** use component instantiation statement ***
exercise_1  u0  (.reset(t_reset),.clk(t_clk),.load(t_load),.data_in(t_data_in),
                 .lamp_ctl(lamp_show));

endmodule



