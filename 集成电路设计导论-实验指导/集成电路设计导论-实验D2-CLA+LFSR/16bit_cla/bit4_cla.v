//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  bit4_cla  (ain,bin,cin,sum,cout,cas_p,cas_g);
  
input   [3:0] ain,bin;
input         cin;

output  [3:0] sum;
output        cout;
output        cas_p,cas_g;

//************************************************

//*** connect with full_adder ***
wire  [3:0] c;
wire  [3:0] p,g;

full_adder  b3  (.a(ain[3]),.b(bin[3]),.ci(c[2]),.p(p[3]),.g(g[3]),.so(sum[3]));

full_adder  b2  (.a(ain[2]),.b(bin[2]),.ci(c[1]),.p(p[2]),.g(g[2]),.so(sum[2]));

full_adder  b1  (.a(ain[1]),.b(bin[1]),.ci(c[0]),.p(p[1]),.g(g[1]),.so(sum[1]));

full_adder  b0  (.a(ain[0]),.b(bin[0]),.ci(cin),.p(p[0]),.g(g[0]),.so(sum[0]));


//*** connect with bit4cla_logic ***

bit4cla_logic cla_logic (.cin(cin),.p3(p[3]),.g3(g[3]),.p2(p[2]),.g2(g[2]),
                         .p1(p[1]),.g1(g[1]),.p0(p[0]),.g0(g[0]),
                         .c3(c[3]),.c2(c[2]),.c1(c[1]),.c0(c[0]));
                         
//************************************************
assign  cout = c[3];

assign  cas_p = &p;
assign  cas_g = g[3] |( g[2] & p[3]) | (g[1] & p[3] & p[2]) | (g[0] & p[3] & p[2] & p[1]);


endmodule
