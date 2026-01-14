//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  full_adder  (a,b,ci,p,g,so);

input   a,b,ci;
output  p,g,so;

//*** calculate the p & g functiom ***
assign  p = a ^ b;
assign  g = a & b;

//*** calculate the sun ***
assign  so = p ^ ci;

endmodule

