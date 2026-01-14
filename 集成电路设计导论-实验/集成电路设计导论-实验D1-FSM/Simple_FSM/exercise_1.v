//***** ASIC Exercise - 1 *****
//*** Counter + Decoder ***
//*** can be used to control colour lamps ***
//*** colour lamps arrangement ***
//*******************
//*** d e f 0 1 2 ***
//*** c         3 ***
//*** b         4 ***
//*** a 9 8 7 6 5 ***
//*******************

module exercise_1 (reset,clk,load,data_in,
  
                   lamp_ctl);
                   
//* input/output ports definition *
input           reset,clk,load;
input [3:0]     data_in;

output  [15:0]  lamp_ctl;
reg     [15:0]  lamp_ctl;

//* in-circuit signals definition *
wire            ena;
reg     [3:0]   cnt1,cnt2;
reg     [15:0]  lamp;

//* circuit RTL description *
//* 4-bit mode-16 counter with sync_data load*
//* used as freq_divider *
always @(posedge reset or posedge clk)
  if (reset)
    cnt1 <= 4'b0000;
  else if (load)
    cnt1 <= data_in;
  else if (cnt1 == 4'b1111)
    cnt1 <= data_in;
  else 
    cnt1 <= cnt1 + 1'b1;
    
assign ena = &cnt1; 

//* 4-bit mode-16 counter *
always @(posedge reset or posedge clk)
  if (reset)
    cnt2 <= 4'b1111;
  else if (ena)
    begin
      if (cnt2 == 4'b0000)
        cnt2 <= 4'b1111;
      else 
        cnt2 <= cnt2 - 1'b1;
    end 
    
//* 4-16 decoder *
always @(reset or cnt2)
  begin
    if (reset)
      lamp = 16'h0000;
    else
      case (cnt2)
        4'b0000 : lamp = 16'b0000_0000_0000_0001;
        4'b0001 : lamp = 16'b0000_0000_0000_0010;
        4'b0010 : lamp = 16'b0000_0000_0000_0100;
        4'b0011 : lamp = 16'b0000_0000_0000_1000;
        4'b0100 : lamp = 16'b0000_0000_0001_0000;
        4'b0101 : lamp = 16'b0000_0000_0010_0000;
        4'b0110 : lamp = 16'b0000_0000_0100_0000;
        4'b0111 : lamp = 16'b0000_0000_1000_0000;
        4'b1000 : lamp = 16'b0000_0001_0000_0000;
        4'b1001 : lamp = 16'b0000_0010_0000_0000;
        4'b1010 : lamp = 16'b0000_0100_0000_0000;
        4'b1011 : lamp = 16'b0000_1000_0000_0000;
        4'b1100 : lamp = 16'b0001_0000_0000_0000;
        4'b1101 : lamp = 16'b0010_0000_0000_0000;
        4'b1110 : lamp = 16'b0100_0000_0000_0000;
        4'b1111 : lamp = 16'b1000_0000_0000_0000;
        default : lamp = 16'b0000_0000_0000_0000;
      endcase
    end
    
//* latch the output control signal *
always @(posedge reset or posedge clk)
  if (reset)
    lamp_ctl <= 16'h0000;
  else
    lamp_ctl <= lamp;
    
endmodule
  
    
  
    
