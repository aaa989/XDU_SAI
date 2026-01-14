// ?? test_cla16bit ????
module  test_cla16bit  ( );

reg [15:0] ain, bin;
reg       cin;
reg       sub;        // ?????????

initial
  begin
    sub = 1'b0;  // ???????
    
    // ????????
    #0    cin = 1'b0;
          ain = 16'd0;
          bin = 16'd0;
    #100  ain = 16'd32767;
          bin = 16'd32768;
    #100  ain = 16'd32767;
          bin = 16'd16384;
          
    // ???????
    #100  sub = 1'b1;
          cin = 1'b0;  // ??? cin ?????? 1
          ain = 16'd100;
          bin = 16'd50;  // 100 - 50 = 50
          
    #100  ain = 16'd50;
          bin = 16'd30;  // 50 - 30 = 20
          
    #100  ain = 16'd100;
          bin = 16'd100;  // 100 - 100 = 0
          
    #100  ain = 16'd50;
          bin = 16'd100;  // 50 - 100 = -50 (????)
          
    // ??????
    #100  sub = 1'b0;
          cin = 1'b1;
          ain = 16'd8;
          bin = 16'd8;
          
    // ????? cin ????????????
    #100  sub = 1'b1;
          cin = 1'b0;  // ??? cin=0????????? 1
          ain = 16'd15;
          bin = 16'd7;  // 15 - 7 = 8
          
    #100  $stop;
  end
  
  //*** connect with circuit to be tested ***
  wire  [15:0] sum_out;
  wire         cout_out;
  
  bit16_cla  u0 (.ain(ain), .bin(bin), .cin(cin), .sum(sum_out), 
                 .cout(cout_out), .sub(sub));
  
  // ????
  initial begin
    $monitor("Time=%t, sub=%b, ain=%d, bin=%d, cin=%b, sum=%d, cout=%b", 
             $time, sub, ain, bin, cin, sum_out, cout_out);
  end
  
endmodule