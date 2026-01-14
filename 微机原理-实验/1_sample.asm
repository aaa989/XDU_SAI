TITLE 8086 Code Template (for EXE file) 
 
; AUTHOR WXT 
; DATE 2024.9.11 
; VERSION 1.00 
; FILE ?.ASM 
 
; 8086 Code Template 
 
; Directive to make EXE output: 
 #MAKE_EXE# 
 
DSEG SEGMENT 'DATA' 
; TODO: add your data here!!!! 
 
 
string_fun db 'Xidian University 2024'
 db '$' 
 
string_origin db 'The original data is: $' 
 
 
string_main db 'please input the function number(1~5)$' 
string_error db 'Wrong number, please input again: $' 
string_broken db 'The pro is broken, Ple. run again $' 
 
 
string_fun1_result db 100 dup(0)
db '$' 
string_fun1_disresult db 'The Upper Case is: $' 
 
string_fun2 db 'XidianUniversity2024',' ','3',' ','6','$' 
string_fun2_result db 'The maximum is: $' 
 
 
 
string_fun3_disp_result db 'The sorted data is: $' 
string_fun3_hex db '134543792' 
 db '$' 
string_fun3_temp dw ? 
 db '$' 
string_fun3_result db 100 dup(0) 
 db '$' 
 
 
string_fun4_info db 'please press anykey to display the time $' 
string_fun4 db 'Now the time is:$' 
string_time db 2 dup(0) 
 db 0DH, 0AH, ':', 0DH, 0AH
 db 2 dup(0) 
 db 0DH, 0AH, ':', 0DH, 0AH
 db 2 dup(0) 
 db '$' 
 
;string_input_character db 'please input character $' 
;string_input_error db 'error please input 0~9 a~z A~Z $' 
;string_fun3_info db 'please input the decimal number $' 
;string_fun3_error db 'input error,please input 0~255 $' 
;string_fun3_hex db 100 dup(?) 
; db '$' 
;string_fun3_temp dw ? 
; db '$' 
;string_fun3_result db 100 dup(?) 
; db '$' 
;string_hexcopy db 100 dup(?) 
; db '$' 
;string_fun4_info db 'please press anykey to display the time $' 
 
DSEG ENDS 
 
SSEG SEGMENT STACK 'STACK' 
 DW 100h DUP(?) 
TOP LABEL WORD 
SSEG ENDS 
 
 
 
CSEG SEGMENT 'CODE' 
 ASSUME CS:CSEG, DS:DSEG, ES:DSEG, SS:SSEG 
;******************************************* 
 
 
 
 
START:  
; set segment registers: 
 MOV AX, DSEG 
 MOV DS, AX 
 MOV ES, AX 
 MOV AX, SSEG 
 MOV SS, AX 
 LEA SP, TOP 
 
 
; TODO: add your code here!!!! 
 MAIN_FUNCTION: 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 LEA DX, string_main 
 MOV AH, 09H 
 INT 21H 
 
 MOV AH, 01H 
 INT 21H 
 
 CMP AL, 31H ; 分析是不是 1-5  
 JB DISP_INPUT_ERR 
 CMP AL, 35H 
 JA DISP_INPUT_ERR 
 JMP FUNC_SEL 
 DISP_INPUT_ERR: 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 LEA DX, string_error 
 MOV AH, 09H 
 INT 21H 
 JMP MAIN_FUNCTION 
 
 FUNC_SEL: 
 CMP AL, 32H 
 JB FUNC_1 
 JE FUNC_2 
 JMP FUNC_345 
 
 FUNC_345: 
 CMP AL, 34H 
 JB FUNC_3 
 JE FUNC_4 
 JMP FUNC_5 
 
  FUNC_1: 
 CALL FUNCTION_1 
 JMP MAIN_FUNCTION 
 
 FUNC_2: 
 CALL FUNCTION_2 
 JMP MAIN_FUNCTION 
 
 FUNC_3: 
 CALL FUNCTION_3 
 JMP MAIN_FUNCTION 
 
 FUNC_4: 
 CALL FUNCTION_4 
 JMP MAIN_FUNCTION 
 
 FUNC_5: 
 CALL FUNCTION_5 
 JMP MAIN_FUNCTION 
 
 ; INPUT YOUR FUNCTION 1 HERE 
 FUNCTION_1 PROC NEAR 
 LEA SI, string_fun 
 LEA DI, string_fun1_result 
 
COUNTER: 
 LODSB 

 CMP AL, '$'  
 JE DONE 
 

 CMP AL, 'a' 
 JB NO_CHANGE 
 CMP AL, 'z' 
 JA NO_CHANGE 
 
 
 SUB AL, 20H 
 
NO_CHANGE: 
 STOSB 
 JMP COUNTER 
 
DONE: 
 
 MOV [DI], '$' 

 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 LEA DX, string_origin  
 MOV AH, 09H 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 LEA DX, string_fun 
 MOV AH, 09H 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 LEA DX, string_fun1_disresult 
 MOV AH, 09H 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH  
 INT 21H 
 
 LEA DX, string_fun1_result 
 MOV AH, 09H 
 INT 21H 
 
 MOV AX,0 
 MOV BX,0 
 MOV CX,0 
 MOV DX,0 
 RET 
FUNCTION_1 ENDP 
 
 ; INPUT YOUR FUNCTION 2 HERE 
 FUNCTION_2 PROC NEAR 
 
 LEA SI, string_fun2 
 MOV CX, 0 
 
LENGTH: 
 LODSB 
 CMP AL, '$' 
 JE DONE_CALCULATE 
 
 
 INC CX 
 JMP LENGTH 
 
DONE_CALCULATE: 
 LEA SI, string_fun2 
 MOV DL, [SI] 
 INC SI 
 DEC CX 
LCMP: 
 CMP DL, [SI] 
 JA NOCHNG 
 MOV DL, [SI] 
 
NOCHNG: 
 INC SI 
 LOOP LCMP ; 找出字符串中最大字符，放入 DL 
 MOV BL,DL 
DISPLAY: 

 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 
 LEA DX, string_origin 
 MOV AH, 09H  
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 
 LEA DX, string_fun2 
 MOV AH, 09H 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0DH 
 INT 21H 
 MOV AH, 02H 
 MOV DL, 0AH 
 INT 21H 
 
 LEA DX, string_fun2_result 
 MOV AH, 09H 
 INT 21H 
 
 MOV DL, BL 
 MOV AH, 02H 
 INT 21H 
 
 MOV AX,0 
 MOV BX,0 
 MOV CX,0 
 MOV DX,0 
 
 RET 
 FUNCTION_2 ENDP 
 
; INPUT YOUR FUNCTION 3 HERE 
 FUNCTION_3 PROC NEAR 
 
 
 MOV AH,2 
 MOV DL,0DH 
 INT 21H 
 MOV AH,2 
 MOV DL,0AH 
 INT 21H 
 LEA DX, string_origin 
 MOV AH,09H 
 INT 21H 
 
 
 LEA SI, string_fun3_hex 
 MOV CX, 0 
 
 CALCULATE: 
 LODSB  
 CMP AL, '$' 
 JE END 
 INC CX 
 JMP CALCULATE 
 
 END: 
 LEA SI, string_fun3_hex 
 PUSH CX 
 MOV BX,CX 
 MOV AX,0 
 
 ORIGION: 
 MOV AH,02H 
 MOV DL,[SI] 
 INT 21H 
 INC SI 
 CMP CX,1 
 JE PRE 
 MOV DL,',' 
 INT 21H 
 LOOP ORIGION 
 
 ; 排序
 PRE: 
 POP CX 
 DEC CX 
 LEA SI, string_fun3_hex 
 ADD SI, CX 
 ; SI 指向末尾字符  
 ; 冒泡排序 
 LP1: 
 PUSH CX 
 PUSH SI 
 LP2: 
 MOV AL,[SI] 
 CMP [SI-1],AL 
 JAE NOXCHG 
 XCHG AL,[SI-1] 
 MOV [SI],AL 
 NOXCHG: 
 DEC SI 
 LOOP LP2 
 POP SI 
 POP CX 
 LOOP LP1 
 
 MOV AX,0 

 MOV AH,2 
 MOV DL,0DH 
 INT 21H 
 MOV AH,2 
 MOV DL,0AH 
 INT 21H 
 LEA DX, string_fun3_disp_result 
 MOV AH,09H 
 INT 21H  
 LEA SI,string_fun3_hex 
 MOV CX,BX 
 MOV AX,0 
 
 RESULT: 
 MOV AH,02H  
 MOV DL,[SI] 
 INT 21H 
 INC SI 
 CMP CX,1 
 JE EXIT 
 MOV DL,',' 
 INT 21H 
 LOOP RESULT 
 
 EXIT: 
 MOV AX,0 
 MOV BX,0 
 MOV CX,0 
 MOV DX,0 
 
 RET 
 FUNCTION_3 ENDP 
 
; INPUT YOUR FUNCTION 4 HERE 
 FUNCTION_4 PROC NEAR 
  
 MOV AH,2 
 MOV DL,0DH 
 INT 21H 
 MOV AH,2 
 MOV DL,0AH 
 INT 21H 
 LEA DX, string_fun4_info 
 MOV AH,09H 
 INT 21H 
 
 MOV AH,01H 
 INT 21H 
 ;获取系统时间
 MOV AH,00H 
 INT 1AH 
 
 PUSH DX 
 PUSH CX 
 
 MOV AH,2 
 MOV DL,0DH 
 INT 21H 
 MOV AH,2 
 MOV DL,0AH 
 INT 21H  
 LEA DX, string_fun4 
 MOV AH,09H 
 INT 21H 
 
 POP CX 
 ; CL 显示为时钟 
 MOV AH,0 
 MOV AL,CL 
 MOV BL,10 
 DIV BL 
 MOV BL,AH 
 ADD AL,30H 
 ADD BL,30H 
 MOV AH,02H  
 MOV DL,AL 
 INT 21H 
 MOV AH,02H 
 MOV DL,BL 
 INT 21H 
 MOV AH,02H 
 MOV DL,':' 
 INT 21H 
  
 POP DX 
 
 ; 获取分钟秒钟 
 MOV CX,DX 
 MOV DX,0  
 MOV AX,0 
 MOV AX,CX 
 MOV BX,18 
 DIV BX  
 MOV DX,0 
 MOV BL,60 
 DIV BL 
 
 ; 显示分钟数 
 MOV CX,0 
 MOV CH,AH 
 MOV CL,AL 
 MOV AH,0 
 MOV AL,CL 
 MOV BL,10 
 DIV BL 
 MOV BL,AH 
 ADD AL,30H 
 ADD BL,30H 
 
 MOV AH,02H 
 MOV DL,AL 
 INT 21H 
 MOV AH,02H 
 MOV DL,BL 
 INT 21H 
 MOV AH,02H 
 MOV DL,':' 
 INT 21H  
 ; 显示秒钟数 
 MOV DX,0 
 MOV AX,0 
 MOV BX,0 
 MOV BL,10 
 MOV AL,CH 
 DIV BL ; 
 MOV BL,AH 
 ADD AL,30H 
 ADD BL,30H 
 
 MOV AH,02H 
 MOV DL,AL 
 INT 21H 
 MOV AH,02H 
 MOV DL,BL 
 INT 21H 
 
 
 MOV AX,0 
 MOV BX,0 
 MOV CX,0 
 MOV DX,0 
 RET 
 FUNCTION_4 ENDP 
 
 
  
; INPUT YOUR FUNCTION 5HERE 
 FUNCTION_5 PROC NEAR 
 
 JMP MAIN_FUNCTION 
 
 RET 
 FUNCTION_5 ENDP 
 
 
 
 
 HLT 
 CSEG ENDS 
 
 END START ; set entry point.