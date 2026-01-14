//************************************************
//*** first generated on 28th,may,2017 ***********
//************************************************

module  bit16_mult  (arst,srst,clk,
                     opa,opb,op_ld,
                     
                     mult_out);

input           arst,srst,clk;
input   [15:0]  opa,opb;
input           op_ld;

output  [31:0]  mult_out;
reg     [31:0]  mult_out;

//************************************************

//*** generate calculate enanle signal ***
reg   cal_enb;
wire  cal_end;

always @(posedge clk or negedge arst)
  if (~arst)
    cal_enb <= 1'b0;
  else if (srst || cal_end)
    cal_enb <= 1'b0;
  else if (op_ld)
    cal_enb <= 1'b1;

//*** shift - add cycle control ***
reg [3:0] cal_cnt;

always @(posedge clk or negedge arst)
  if (~arst)
    cal_cnt <= 4'h0;
  else if (srst || op_ld)
    cal_cnt <= 4'h0;
  else if (cal_enb)
    cal_cnt <= cal_cnt + 1'b1;
    
assign  cal_end = &cal_cnt;
    
//*** partial-product select ***
wire  [15:0]  par_product;
reg   [15:0]  multiplicand;

assign par_product = (mult_out[0] == 1'b1) ? multiplicand : 16'h0000;

//*** sum of partial-product ***
wire  [16:0] sum_pproduct;

assign  sum_pproduct = par_product + mult_out[31:16];

//*** operands loda-in & shift operation ***


always @(posedge clk or negedge arst)
  if (~arst)
    begin
      multiplicand <= 16'h0000;
      mult_out <= 32'h00000000;
    end
  else if (srst)
    begin
      multiplicand <= 16'h0000;
      mult_out <= 32'h00000000;
    end
  else if (op_ld)
    begin
      multiplicand <= opa;
      mult_out <= {16'h0000,opb};
    end
  else if (cal_enb)
    mult_out <= ({sum_pproduct,mult_out[15:0]}) >> 1;

endmodule


//================================================
//*** ?????4-bit?????? *****************
//================================================

module  bin_cnt_4bit  (
  input   async_rst,
  input   sync_rst,
  input   clk,
  input   enb,
  input   [3:0] d_in,
  input   d,
  
  output  [3:0] cnt_out,
  output  cout
);

reg [3:0] cnt;
  
always @(posedge clk or posedge async_rst)
  if (async_rst)
    cnt <= 4'b0000;
  else if (sync_rst)
    cnt <= 4'b0000;
  else if (d)
    cnt <= d_in;
  else if (enb)
    cnt <= cnt + 4'b0001;

assign cnt_out = cnt;
assign cout = (cnt == 4'b1111) ? 1'b1 : 1'b0;

endmodule