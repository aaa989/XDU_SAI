//************************************************
//*** Modified bit16_cla with Add/Sub support ****
//************************************************

module bit16_cla_add_sub (
    input [15:0] ain, bin,
    input cin,
    input add_sub,      // ????????
    output [15:0] sum,
    output cout
);

//*** ?????? ***
wire [15:0] bin_actual;
wire cin_actual;

//*** ??????? ***
assign bin_actual = add_sub ? ~bin : bin;      // ?????
assign cin_actual = add_sub ? 1'b1 : cin;      // ???cin=1

//*** ??4?4?CLA??????? ***
wire p15_12, p11_8, p7_4, p3_0;
wire g15_12, g11_8, g7_4, g3_0;
wire c11, c7, c3;

//*** ???4?4?CLA?? ***
bit4_cla t3 (
    .ain(ain[15:12]), 
    .bin(bin_actual[15:12]), 
    .cin(c11), 
    .sum(sum[15:12]), 
    .cas_p(p15_12), 
    .cas_g(g15_12)
);

bit4_cla t2 (
    .ain(ain[11:8]), 
    .bin(bin_actual[11:8]), 
    .cin(c7), 
    .sum(sum[11:8]), 
    .cas_p(p11_8), 
    .cas_g(g11_8)
);

bit4_cla t1 (
    .ain(ain[7:4]), 
    .bin(bin_actual[7:4]), 
    .cin(c3), 
    .sum(sum[7:4]), 
    .cas_p(p7_4), 
    .cas_g(g7_4)
);

bit4_cla t0 (
    .ain(ain[3:0]), 
    .bin(bin_actual[3:0]), 
    .cin(cin_actual),  // ??????cin
    .sum(sum[3:0]), 
    .cas_p(p3_0), 
    .cas_g(g3_0)
);

//*** ?????? ***
bit4cla_logic cla_logic (
    .cin(cin_actual),   // ??????cin
    .p3(p15_12), .g3(g15_12),
    .p2(p11_8), .g2(g11_8),
    .p1(p7_4), .g1(g7_4),
    .p0(p3_0), .g0(g3_0),
    .c3(cout), .c2(c11), .c1(c7), .c0(c3)
);

endmodule