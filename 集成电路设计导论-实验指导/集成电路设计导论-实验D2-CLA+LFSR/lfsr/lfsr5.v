//**********************************************
//*** first generated on 28th,may,2017 *********
//**********************************************
//*** polynomial: x^4 + x^3 + 1 ****************

module  lfsr (async_rst, sync_rst, clk, cnt_out);

input         async_rst, sync_rst;
input         clk;

output  [3:0] cnt_out;      // ??4???
reg     [3:0] cnt_out;      // ??4????

always @(posedge clk or negedge async_rst)
  if (~async_rst)
    cnt_out <= 4'b0000;     // ????4?
  else if (sync_rst)
    cnt_out <= 4'b1111;     // ?????4??1
  else
    begin
      // 4?LFSR: x^4 + x^3 + 1
      // ??? = cnt_out[3] ^ cnt_out[2]
      cnt_out[0] <= cnt_out[3] ^ cnt_out[2];  // ????
      cnt_out[1] <= cnt_out[0];
      cnt_out[2] <= cnt_out[1];
      cnt_out[3] <= cnt_out[2];      
      // ?4????
    end
    
endmodule