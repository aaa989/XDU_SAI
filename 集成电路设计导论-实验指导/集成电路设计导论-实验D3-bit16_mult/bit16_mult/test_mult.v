`timescale 1ns/1ns

//************************************************
//*** first generated on 28th,may,2017 ***********
//***
//***
//************************************************

module  test_mult  ( );

reg        t_arst,t_srst,t_clk;
reg [15:0] t_opa,t_opb;
reg        t_op_ld;

//*** t_clk ***
initial
  begin 
    t_clk = 1'b0;
  end
   
always #50 t_clk = ~t_clk;

//*** t_arst ***
initial
	begin
  		#0    t_arst = 1'b1;
		#50   t_arst = 1'b0;
		#20   t_arst = 1'b1;  
   end
   
//*** t_srst ***
initial
	begin
  		#0    t_srst = 1'b0;
		#200  t_srst = 1'b1;
		#100   t_srst = 1'b0;  
   end


//*** t_op_ld,t_opa,t_opb ***
initial
  begin
    #1      t_op_ld = 1'b0;
            t_opa = 16'd0;
            t_opb = 16'd0;
    #100    t_op_ld = 1'b0;
    
    #450    t_op_ld = 1'b1;
            t_opa = 16'd0;
            t_opb = 16'd0;
    #100    t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd1;
            t_opb = 16'd4;
    #100    t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd9;
            t_opb = 16'd15;
    #100    t_op_ld = 1'b0;

    #2000   t_op_ld = 1'b1;
            t_opa = 16'd31;
            t_opb = 16'd63;
    #100    t_op_ld = 1'b0; 

    #2000   t_op_ld = 1'b1;
            t_opa = 16'd255;
            t_opb = 16'd512;
    #100    t_op_ld = 1'b0;        

    #2000   t_op_ld = 1'b1;
            t_opa = 16'd1023;
            t_opb = 16'd2048;
    #100    t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd4095;
            t_opb = 16'd8195;
    #100    t_op_ld = 1'b0;          

    #2000   t_op_ld = 1'b1;
            t_opa = 16'd32767;
            t_opb = 16'd32768;
    #100     t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd32768;
            t_opb = 16'd65535;
    #100     t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd40956;
            t_opb = 16'd47281;
    #100     t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd65535;
            t_opb = 16'd65534;
    #100    t_op_ld = 1'b0;
    
    #2000   t_op_ld = 1'b1;
            t_opa = 16'd65535;
            t_opb = 16'd65535;
    #100    t_op_ld = 1'b0;
  end
  
  //*** connect with circuit to be tested ***
  wire  [31:0] product_out;
  
  bit16_mult  mult  (.arst(t_arst),.srst(t_srst),.clk(t_clk),
                     .opa(t_opa),.opb(t_opb),.op_ld(t_op_ld),
                     
                     .mult_out(product_out));
  
endmodule