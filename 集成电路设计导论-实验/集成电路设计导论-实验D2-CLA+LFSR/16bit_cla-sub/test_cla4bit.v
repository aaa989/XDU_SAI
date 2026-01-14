//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  test_cla4bit  ( );

reg [3:0] ain,bin;
reg       cin;

initial
  begin
    #0    cin = 1'b0;
          ain = 4'd0;
          bin = 4'd0;
    #100  ain = 4'd1;
          bin = 4'd2;
    #100  ain = 4'd2;
          bin = 4'd3;
    #100  ain = 4'd3;
          bin = 4'd4;
    #100  ain = 4'd4;
          bin = 4'd5;
    #100  cin = 1'b1;
          ain = 4'd5;
          bin = 4'd6;
    #100  cin = 1'b0;
          ain = 4'd6;
          bin = 4'd7;
    #100  ain = 4'd7;
          bin = 4'd8;
    #100  cin = 1'b1;
          ain = 4'd7;
          bin = 4'd8;
    #100  ain = 4'd0;
          bin = 4'd1;
    #100  ain = 4'd1;
          bin = 4'd1;
    #100  ain = 4'd2;
          bin = 4'd2;
    #100  ain = 4'd3;
          bin = 4'd3;
    #100  cin = 1'b0;
          ain = 4'd4;
          bin = 4'd4;
    #100  cin = 1'b1;
          ain = 4'd5;
          bin = 4'd5;
    #100  ain = 4'd6;
          bin = 4'd6;
    #100  cin = 1'b1;
          ain = 4'd7;
          bin = 4'd7;
    #100  cin = 1'b0;
          ain = 4'd8;
          bin = 4'd8;
    #100  cin = 1'b1;
          ain = 4'd8;
          bin = 4'd8;
    #100  cin = 1'b1;
          ain = 4'd8;
          bin = 4'd9;
  end
  
  //*** connect with circuit to be tested ***
  wire  [3:0] sum_out;
  wire        cout_out;
  
  bit4_cla  u0 (.ain(ain),.bin(bin),.cin(cin),.sum(sum_out),.cout(cout_out));
  
endmodule
