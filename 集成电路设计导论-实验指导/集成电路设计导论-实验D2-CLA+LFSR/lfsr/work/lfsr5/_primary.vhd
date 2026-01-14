library verilog;
use verilog.vl_types.all;
entity lfsr5 is
    port(
        async_rst       : in     vl_logic;
        sync_rst        : in     vl_logic;
        clk             : in     vl_logic;
        cnt_out         : out    vl_logic_vector(4 downto 0)
    );
end lfsr5;
