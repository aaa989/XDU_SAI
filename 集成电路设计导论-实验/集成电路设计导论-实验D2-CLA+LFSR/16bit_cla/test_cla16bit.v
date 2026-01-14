//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  test_cla16bit  ( );

reg [15:0] ain,bin;
reg       cin;

initial
  begin
    #0    cin = 1'b0;
          ain = 16'd0;
          bin = 16'd0;
    #100  ain = 16'd32767;
          bin = 16'd32768;
    #100  ain = 16'd32767;
          bin = 16'd16384;
    #100  ain = 16'd32767;
          bin = 16'd8192;
    #100  ain = 16'd32767;
          bin = 16'd4096;
    #100  cin = 1'b1;
          ain = 16'd32767;
          bin = 16'd32768;
    #100  cin = 1'b0;
          ain = 16'd16384;
          bin = 16'd8193;
    #100  ain = 16'd8192;
          bin = 16'd4095;
    #100  cin = 1'b1;
          ain = 16'd4096;
          bin = 16'd2048;
    #100  ain = 16'd2048;
          bin = 16'd1023;
    #100  ain = 16'd1024;
          bin = 16'd511;
    #100  ain = 16'd511;
          bin = 16'd256;
    #100  ain = 16'd255;
          bin = 16'd128;
    #100  cin = 1'b0;
          ain = 16'd127;
          bin = 16'd63;
    #100  cin = 1'b1;
          ain = 16'd64;
          bin = 16'd32;
    #100  ain = 16'd31;
          bin = 16'd16;
    #100  cin = 1'b1;
          ain = 16'd15;
          bin = 16'd8;
    #100  cin = 1'b0;
          ain = 16'd8;
          bin = 16'd8;
    #100  cin = 1'b1;
          ain = 16'd8;
          bin = 16'd8;
    #100  cin = 1'b1;
          ain = 16'd8;
          bin = 16'd9;
  end
  
  //*** connect with circuit to be tested ***
  wire  [15:0] sum_out;
  wire         cout_out;
  
  bit16_cla  u0 (.ain(ain),.bin(bin),.cin(cin),.sum(sum_out),.cout(cout_out));
  
endmodule

