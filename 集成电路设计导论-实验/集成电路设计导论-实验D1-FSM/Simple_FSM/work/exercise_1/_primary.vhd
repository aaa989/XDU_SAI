library verilog;
use verilog.vl_types.all;
entity exercise_1 is
    port(
        reset           : in     vl_logic;
        clk             : in     vl_logic;
        load            : in     vl_logic;
        data_in         : in     vl_logic_vector(3 downto 0);
        lamp_ctl        : out    vl_logic_vector(15 downto 0)
    );
end exercise_1;
