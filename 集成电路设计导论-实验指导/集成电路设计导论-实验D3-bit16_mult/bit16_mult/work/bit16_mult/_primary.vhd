library verilog;
use verilog.vl_types.all;
entity bit16_mult is
    port(
        arst            : in     vl_logic;
        srst            : in     vl_logic;
        clk             : in     vl_logic;
        opa             : in     vl_logic_vector(15 downto 0);
        opb             : in     vl_logic_vector(15 downto 0);
        op_ld           : in     vl_logic;
        mult_out        : out    vl_logic_vector(31 downto 0)
    );
end bit16_mult;
