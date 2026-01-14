//************************************************
//*** first generated on 27th,may,2017 ***********
//***
//***
//************************************************

module  bit4cla_logic  (cin,p3,g3,p2,g2,p1,g1,p0,g0,
                  
                       c3,c2,c1,c0);
                       
input   cin;
input   p3,g3,p2,g2,p1,g1,p0,g0;

output  c3,c2,c1,c0;

//************************************************

assign  c3 = g3 | g2 & p3 | g1 & p3 & p2 | g0 & p3 & p2 & p1 | cin & p3 & p2 & p1 & p0;
assign  c2 = g2 | g1 & p2 | g0 & p2 & p1 | cin & p2 & p1 & p0; 
assign  c1 = g1 | g0 & p1 | cin & p1 & p0; 
assign  c0 = g0 | cin & p0;

endmodule

