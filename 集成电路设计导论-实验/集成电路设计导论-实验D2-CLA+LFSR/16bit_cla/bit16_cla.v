//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  bit16_cla  (ain,bin,cin,sum,cout);

input   [15:0]  ain,bin;
input           cin;

output  [15:0]  sum;
output          cout;

//************************************************

//*** connect with bit4_cla ***

wire  p15_12,p11_8,p7_4,p3_0;
wire  g15_12,g11_8,g7_4,g3_0;
wire  c11,c7,c3;

bit4_cla  t3 (.ain(ain[15:12]),.bin(bin[15:12]),.cin(c11),.sum(sum[15:12]),.cas_p(p15_12),.cas_g(g15_12));

bit4_cla  t2 (.ain(ain[11:8]),.bin(bin[11:8]),.cin(c7),.sum(sum[11:8]),.cas_p(p11_8),.cas_g(g11_8));

bit4_cla  t1 (.ain(ain[7:4]),.bin(bin[7:4]),.cin(c3),.sum(sum[7:4]),.cas_p(p7_4),.cas_g(g7_4));

bit4_cla  t0 (.ain(ain[3:0]),.bin(bin[3:0]),.cin(cin),.sum(sum[3:0]),.cas_p(p3_0),.cas_g(g3_0));

//*** connect with bit4cla_logic ***

bit4cla_logic  cla_logic  (.cin(cin),.p3(p15_12),.g3(g15_12),.p2(p11_8),.g2(g11_8),
                           .p1(p7_4),.g1(g7_4),.p0(p3_0),.g0(g3_0),
                           .c3(cout),.c2(c11),.c1(c7),.c0(c3));
                           
endmodule
