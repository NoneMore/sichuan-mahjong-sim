.seh_proc _RNvMNtNtCs4DgSkW5ScxR_19sichuan_mahjong_sim6policy8analysisNtB2_17DistanceEvaluator7deficit
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$1848, %rsp
	.seh_stackalloc 1848
	movdqa	%xmm7, 1824(%rsp)
	.seh_savexmm %xmm7, 1824
	movaps	%xmm6, 1808(%rsp)
	.seh_savexmm %xmm6, 1808
	.seh_endprologue
	movq	%rdx, %r11
	movq	%rcx, %r10
	cmpq	$0, 72(%rcx)
	movq	%rdx, 80(%rsp)
	movq	%rcx, 48(%rsp)
	je	.LBB19_10
	movq	80(%r10), %rcx
	movq	88(%r10), %rdx
	movq	%r11, %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %r10
	movq	%rax, %r8
	shrq	$57, %r8
	movq	48(%r10), %rcx
	movq	56(%r10), %rdx
	movd	%r8d, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	xorl	%r8d, %r8d
	pcmpeqd	%xmm1, %xmm1
.LBB19_2:
	andq	%rdx, %rax
	movdqu	(%rcx,%rax), %xmm2
	movdqa	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb	%xmm3, %r9d
	testl	%r9d, %r9d
	je	.LBB19_5
.LBB19_3:
	rep		bsfl	%r9d, %r11d
	addq	%rax, %r11
	andq	%rdx, %r11
	movq	%r11, %r10
	negq	%r10
	leaq	(%r10,%r10,8), %r10
	leaq	(%r10,%r10,2), %r10
	subq	%r11, %r10
	movq	80(%rsp), %r11
	movdqu	(%r11), %xmm3
	movdqu	11(%r11), %xmm4
	movdqu	-28(%rcx,%r10), %xmm5
	pcmpeqb	%xmm3, %xmm5
	movdqu	-17(%rcx,%r10), %xmm3
	pcmpeqb	%xmm4, %xmm3
	pand	%xmm5, %xmm3
	pmovmskb	%xmm3, %r11d
	cmpl	$65535, %r11d
	je	.LBB19_130
	leal	-1(%r9), %r10d
	andw	%r9w, %r10w
	movl	%r10d, %r9d
	movq	48(%rsp), %r10
	jne	.LBB19_3
	.p2align	4
.LBB19_5:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	movq	80(%rsp), %r11
	jne	.LBB19_10
	addq	%r8, %rax
	addq	$16, %rax
	addq	$16, %r8
	jmp	.LBB19_2
.LBB19_10:
	movq	$-1, 89(%rsp)
	movb	$-1, 97(%rsp)
	movb	$0, 88(%rsp)
	movq	$0, 264(%rsp)
	movq	$3, 272(%rsp)
	movw	$256, 280(%rsp)
	movb	$2, 282(%rsp)
	leaq	120(%r10), %rax
	movq	%rax, 192(%rsp)
	pcmpeqd	%xmm7, %xmm7
	leaq	32(%r10), %rax
	movq	%rax, 184(%rsp)
	movaps	__xmm@ffffffffffffffff0000000000000000(%rip), %xmm6
	xorl	%r14d, %r14d
	jmp	.LBB19_11
	.p2align	4
.LBB19_112:
	movzwl	560(%rsp), %eax
	movw	%ax, 96(%rsp)
	movq	552(%rsp), %rax
	movq	%rax, 88(%rsp)
	cmpq	$3, %r14
	movq	80(%rsp), %r11
	movq	48(%rsp), %r10
	je	.LBB19_7
.LBB19_11:
	movq	%r14, 120(%rsp)
	movzbl	280(%rsp,%r14), %eax
	movb	%al, 47(%rsp)
	movzbl	%al, %eax
	leaq	(%rax,%rax,8), %r8
	movzbl	8(%r11,%r8), %ecx
	movb	%cl, 112(%rsp)
	movq	(%r11,%r8), %rcx
	movq	%rcx, 104(%rsp)
	movzbl	8(%r11,%r8), %ecx
	leaq	135(%rsp), %rdx
	movb	%cl, 8(%rdx)
	movq	%r8, %r14
	movq	(%r11,%r8), %rcx
	movq	%rcx, (%rdx)
	movb	%al, 134(%rsp)
	cmpq	$0, 24(%r10)
	movq	%r10, %rsi
	je	.LBB19_20
	movq	32(%rsi), %rcx
	movq	40(%rsi), %rdx
	leaq	134(%rsp), %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRThAhj9_EECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	%rax, %r8
	shrq	$57, %r8
	movq	(%rsi), %rcx
	movq	8(%rsi), %rdx
	movd	%r8d, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	movzbl	143(%rsp), %r8d
	movq	135(%rsp), %r9
	xorl	%r10d, %r10d
.LBB19_13:
	andq	%rdx, %rax
	movdqu	(%rcx,%rax), %xmm1
	movdqa	%xmm1, %xmm2
	pcmpeqb	%xmm0, %xmm2
	pmovmskb	%xmm2, %r11d
	testl	%r11d, %r11d
	je	.LBB19_17
.LBB19_14:
	rep		bsfl	%r11d, %esi
	addq	%rax, %rsi
	andq	%rdx, %rsi
	negq	%rsi
	leaq	(%rsi,%rsi,4), %rsi
	movzbl	47(%rsp), %ebx
	cmpb	-20(%rcx,%rsi,4), %bl
	jne	.LBB19_16
	movzbl	-11(%rcx,%rsi,4), %edi
	movq	-19(%rcx,%rsi,4), %rbx
	xorq	%r9, %rbx
	xorl	%r8d, %edi
	movzbl	%dil, %edi
	orq	%rbx, %rdi
	je	.LBB19_19
.LBB19_16:
	leal	-1(%r11), %esi
	andw	%r11w, %si
	movl	%esi, %r11d
	jne	.LBB19_14
	.p2align	4
.LBB19_17:
	pcmpeqb	%xmm7, %xmm1
	pmovmskb	%xmm1, %r11d
	testl	%r11d, %r11d
	movq	80(%rsp), %r11
	movq	48(%rsp), %rsi
	jne	.LBB19_20
	addq	%r10, %rax
	addq	$16, %rax
	addq	$16, %r10
	jmp	.LBB19_13
	.p2align	4
.LBB19_20:
	leaq	(%r11,%r14), %rax
	movq	%rax, 200(%rsp)
	addq	192(%rsp), %r14
	movq	%r14, 224(%rsp)
	movq	112(%rsi), %rsi
	leaq	1559(%rsp), %rax
	movdqu	%xmm7, 233(%rax)
	movdqu	%xmm7, 224(%rax)
	movdqu	%xmm7, 208(%rax)
	movdqu	%xmm7, 192(%rax)
	movdqu	%xmm7, 176(%rax)
	movdqu	%xmm7, 160(%rax)
	movdqu	%xmm7, 144(%rax)
	movdqu	%xmm7, 128(%rax)
	movdqu	%xmm7, 112(%rax)
	movdqu	%xmm7, 96(%rax)
	movdqu	%xmm7, 80(%rax)
	movdqu	%xmm7, 64(%rax)
	movdqu	%xmm7, 48(%rax)
	movdqu	%xmm7, 32(%rax)
	movdqu	%xmm7, 16(%rax)
	movdqu	%xmm7, (%rax)
	movl	$498, %r8d
	leaq	1060(%rsp), %rcx
	xorl	%edx, %edx
	callq	memset
	movb	$0, 1558(%rsp)
	movw	$0, 1058(%rsp)
	movl	$1, %ebx
	xorl	%edi, %edi
	movq	%rsi, 216(%rsp)
	jmp	.LBB19_21
	.p2align	4
.LBB19_28:
	movl	$250, %r8d
	leaq	1558(%rsp), %rcx
	leaq	288(%rsp), %rdx
	callq	memcpy
	movl	$500, %r8d
	leaq	1058(%rsp), %rcx
	leaq	552(%rsp), %rdx
	callq	memcpy
	movq	208(%rsp), %rdi
	testq	%rbx, %rbx
	je	.LBB19_29
.LBB19_21:
	cmpq	$9, %rdi
	je	.LBB19_29
	movdqu	%xmm7, 522(%rsp)
	movdqa	%xmm7, 512(%rsp)
	movdqa	%xmm7, 496(%rsp)
	movdqa	%xmm7, 480(%rsp)
	movdqa	%xmm7, 464(%rsp)
	movdqa	%xmm7, 448(%rsp)
	movdqa	%xmm7, 432(%rsp)
	movdqa	%xmm7, 416(%rsp)
	movdqa	%xmm7, 400(%rsp)
	movdqa	%xmm7, 384(%rsp)
	movdqa	%xmm7, 368(%rsp)
	movdqa	%xmm7, 352(%rsp)
	movdqa	%xmm7, 336(%rsp)
	movdqa	%xmm7, 320(%rsp)
	movdqa	%xmm7, 304(%rsp)
	movdqa	%xmm7, 288(%rsp)
	movl	$500, %r8d
	leaq	552(%rsp), %rcx
	xorl	%edx, %edx
	callq	memset
	cmpq	$251, %rbx
	jae	.LBB19_152
	leaq	1(%rdi), %rax
	movq	%rax, 208(%rsp)
	leaq	(%rsp,%rbx,2), %rax
	addq	$1058, %rax
	movq	%rax, 232(%rsp)
	leaq	1058(%rsp), %rax
	xorl	%ebx, %ebx
	jmp	.LBB19_25
	.p2align	4
.LBB19_24:
	movq	240(%rsp), %rax
	addq	$2, %rax
	cmpq	232(%rsp), %rax
	movq	216(%rsp), %rsi
	movq	56(%rsp), %rbx
	je	.LBB19_28
.LBB19_25:
	movzwl	(%rax), %ecx
	cmpq	$250, %rcx
	jae	.LBB19_70
	movq	%rax, 240(%rsp)
	movq	%rbx, 56(%rsp)
	movzwl	%cx, %eax
	shrl	%eax
	imull	$205, %eax, %edx
	movzwl	%dx, %edx
	shrl	$10, %edx
	leal	(%rdx,%rdx,4), %edx
	subb	%dl, %al
	movzbl	%al, %r9d
	imull	$205, %ecx, %eax
	movzwl	%ax, %eax
	shrl	$11, %eax
	imull	$52, %eax, %edx
	movzbl	%dh, %edx
	leal	(%rdx,%rdx,4), %edx
	subb	%dl, %al
	leal	(%rcx,%rcx,4), %edx
	leal	(%rcx,%rdx,8), %edx
	movzwl	%dx, %edx
	shrl	$11, %edx
	movq	%rsi, %r10
	movq	%r9, 176(%rsp)
	subq	%r9, %r10
	cmpq	%r10, %rsi
	movq	%r10, 168(%rsp)
	cmovbq	%rsi, %r10
	cmpq	$7, %rdi
	movzbl	1558(%rsp,%rcx), %r11d
	movzbl	%al, %eax
	movl	$0, %r8d
	cmovaeq	%r8, %r10
	addb	%al, %dl
	movzbl	%dl, %edx
	movq	%rdx, 160(%rsp)
	leal	(%rax,%rax,4), %eax
	movzbl	%al, %eax
	movq	%rax, 152(%rsp)
	movq	224(%rsp), %rax
	movzbl	(%rax,%rdi), %esi
	testb	$1, %cl
	movq	%r10, 144(%rsp)
	jne	.LBB19_27
	xorl	%ecx, %ecx
	xorl	%eax, %eax
	.p2align	4
.LBB19_45:
	cmpq	%r10, %rcx
	adcq	$0, %rax
	movq	%rax, 248(%rsp)
	movq	168(%rsp), %r9
	subq	%rcx, %r9
	movq	160(%rsp), %rax
	leaq	(%rcx,%rax), %r8
	movq	152(%rsp), %rax
	addq	%rcx, %rax
	leaq	(%rax,%rax,4), %r14
	movq	176(%rsp), %rax
	movq	%rcx, 256(%rsp)
	addq	%rcx, %rax
	addq	%rax, %r14
	xorl	%r12d, %r12d
	xorl	%edx, %edx
	.p2align	4
.LBB19_46:
	cmpq	%r9, %r12
	adcq	$0, %rdx
	leaq	(%r12,%r12,2), %rax
	leaq	(%r8,%rax), %rbx
	leaq	(%r14,%r12), %r10
	addq	%r10, %r10
	leaq	(%r8,%rax), %r13
	addq	$2, %r13
	cmpq	%rsi, %r13
	jbe	.LBB19_47
	cmpq	%rsi, %rbx
	ja	.LBB19_66
	subb	104(%rsp,%rdi), %bl
	movzbl	%bl, %eax
	movl	$0, %ecx
	cmovbl	%ecx, %eax
	addb	%r11b, %al
	movzbl	%al, %eax
	movl	$255, %ecx
	cmovbl	%ecx, %eax
	cmpq	$249, %r10
	ja	.LBB19_61
	movzbl	288(%rsp,%r10), %ecx
	cmpb	%cl, %al
	jae	.LBB19_66
	cmpb	$-1, %cl
	jne	.LBB19_65
	jmp	.LBB19_57
	.p2align	4
.LBB19_47:
	xorl	%ecx, %ecx
	cmpq	%rsi, %rbx
	seta	%cl
	movl	%ebx, %eax
	cmoval	%r13d, %eax
	movzbl	104(%rsp,%rdi), %ebp
	subb	%bpl, %al
	movzbl	%al, %eax
	movl	$0, %r15d
	cmovbl	%r15d, %eax
	addb	%r11b, %al
	movzbl	%al, %eax
	movl	$255, %r15d
	cmovbl	%r15d, %eax
	orq	%r10, %rcx
	cmpq	$249, %r10
	ja	.LBB19_62
	movzbl	288(%rsp,%rcx), %r15d
	cmpb	%r15b, %al
	jae	.LBB19_53
	cmpb	$-1, %r15b
	jne	.LBB19_52
	movq	56(%rsp), %r15
	cmpq	$249, %r15
	ja	.LBB19_58
	movw	%cx, 552(%rsp,%r15,2)
	incq	%r15
	movq	%r15, 56(%rsp)
.LBB19_52:
	movb	%al, 288(%rsp,%rcx)
.LBB19_53:
	cmpq	%rsi, %rbx
	ja	.LBB19_66
	subb	%bpl, %r13b
	movzbl	%r13b, %eax
	movl	$0, %ecx
	cmovbl	%ecx, %eax
	addb	%r11b, %al
	movzbl	%al, %eax
	movl	$255, %ecx
	cmovbl	%ecx, %eax
	movzbl	289(%rsp,%r10), %ecx
	cmpb	%cl, %al
	jae	.LBB19_66
	orq	$1, %r10
	cmpb	$-1, %cl
	jne	.LBB19_65
.LBB19_57:
	movq	56(%rsp), %rcx
	cmpq	$249, %rcx
	ja	.LBB19_58
	movw	%r10w, 552(%rsp,%rcx,2)
	incq	%rcx
	movq	%rcx, 56(%rsp)
.LBB19_65:
	movb	%al, 288(%rsp,%r10)
.LBB19_66:
	cmpq	%r9, %r12
	jae	.LBB19_68
	movq	%rdx, %r12
	cmpq	%r9, %rdx
	jbe	.LBB19_46
.LBB19_68:
	movq	144(%rsp), %r10
	cmpq	%r10, 256(%rsp)
	jae	.LBB19_24
	movq	248(%rsp), %rax
	movq	%rax, %rcx
	cmpq	%r10, %rax
	jbe	.LBB19_45
	jmp	.LBB19_24
	.p2align	4
.LBB19_27:
	xorl	%eax, %eax
	.p2align	4
.LBB19_74:
	movq	%rax, %rdx
	cmpq	%r10, %rax
	adcq	$0, %rax
	movq	168(%rsp), %r8
	subq	%rdx, %r8
	movq	160(%rsp), %rcx
	leaq	(%rdx,%rcx), %r9
	movq	152(%rsp), %rcx
	addq	%rdx, %rcx
	leaq	(%rcx,%rcx,4), %r10
	movq	176(%rsp), %rcx
	addq	%rdx, %rcx
	addq	%rcx, %r10
	xorl	%r14d, %r14d
	xorl	%ebx, %ebx
	.p2align	4
.LBB19_77:
	cmpq	%r8, %r14
	adcq	$0, %rbx
	leaq	(%r14,%r14,2), %r12
	addq	%r9, %r12
	cmpq	%rsi, %r12
	ja	.LBB19_75
	leaq	(%r10,%r14), %r15
	leaq	1(,%r15,2), %rcx
	addq	%r15, %r15
	subb	104(%rsp,%rdi), %r12b
	movzbl	%r12b, %ebp
	movl	$0, %r12d
	cmovbl	%r12d, %ebp
	addb	%r11b, %bpl
	movzbl	%bpl, %ebp
	movl	$255, %r12d
	cmovbl	%r12d, %ebp
	cmpq	$249, %r15
	ja	.LBB19_62
	movzbl	288(%rsp,%rcx), %r15d
	cmpb	%r15b, %bpl
	jae	.LBB19_75
	cmpb	$-1, %r15b
	jne	.LBB19_83
	movq	56(%rsp), %r15
	cmpq	$249, %r15
	ja	.LBB19_58
	movw	%cx, 552(%rsp,%r15,2)
	incq	%r15
	movq	%r15, 56(%rsp)
.LBB19_83:
	movb	%bpl, 288(%rsp,%rcx)
.LBB19_75:
	cmpq	%r8, %r14
	jae	.LBB19_72
	movq	%rbx, %r14
	cmpq	%r8, %rbx
	jbe	.LBB19_77
.LBB19_72:
	movq	144(%rsp), %r10
	cmpq	%r10, %rdx
	jae	.LBB19_24
	cmpq	%r10, %rax
	jbe	.LBB19_74
	jmp	.LBB19_24
	.p2align	4
.LBB19_29:
	movw	$-1, 72(%rsp)
	movq	$-1, 64(%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	48(%rsp), %r8
	.p2align	4
.LBB19_30:
	cmpq	%rsi, %rcx
	adcq	$0, %rdx
	leaq	(%rcx,%rcx), %rax
	cmpq	$249, %rax
	ja	.LBB19_84
	cmpq	$4, %rcx
	ja	.LBB19_86
	movzwl	1558(%rsp,%rcx,2), %eax
	movw	%ax, 64(%rsp,%rcx,2)
	cmpq	%rsi, %rcx
	jae	.LBB19_34
	movq	%rdx, %rcx
	cmpq	%rsi, %rdx
	jbe	.LBB19_30
.LBB19_34:
	movq	200(%rsp), %rdx
	movzbl	8(%rdx), %eax
	leaq	1060(%rsp), %rcx
	movb	%al, 7(%rcx)
	movq	(%rdx), %rax
	movq	%rax, -1(%rcx)
	movzbl	47(%rsp), %eax
	movb	%al, 1058(%rsp)
	movq	32(%r8), %rcx
	movq	40(%r8), %rdx
	leaq	1058(%rsp), %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRThAhj9_EECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rcx
	cmpq	$0, 16(%rcx)
	je	.LBB19_35
.LBB19_36:
	movq	(%rcx), %r13
	movq	8(%rcx), %r8
	movq	%rax, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	movzbl	1067(%rsp), %r10d
	movq	1059(%rsp), %r11
	xorl	%edi, %edi
	xorl	%esi, %esi
.LBB19_37:
	andq	%r8, %rax
	movdqu	(%r13,%rax), %xmm1
	movdqa	%xmm1, %xmm2
	pcmpeqb	%xmm0, %xmm2
	pmovmskb	%xmm2, %ebx
	testl	%ebx, %ebx
	je	.LBB19_41
.LBB19_38:
	rep		bsfl	%ebx, %r14d
	addq	%rax, %r14
	andq	%r8, %r14
	negq	%r14
	leaq	(%r14,%r14,4), %r14
	movzbl	47(%rsp), %ecx
	cmpb	-20(%r13,%r14,4), %cl
	jne	.LBB19_40
	movzbl	-11(%r13,%r14,4), %r15d
	movq	-19(%r13,%r14,4), %r12
	xorq	%r11, %r12
	xorl	%r10d, %r15d
	movzbl	%r15b, %r15d
	orq	%r12, %r15
	je	.LBB19_94
.LBB19_40:
	leal	-1(%rbx), %ebp
	andw	%bx, %bp
	movl	%ebp, %ebx
	jne	.LBB19_38
	.p2align	4
.LBB19_41:
	cmpq	$1, %rdi
	je	.LBB19_88
	pmovmskb	%xmm1, %r9d
	testl	%r9d, %r9d
	je	.LBB19_43
	rep		bsfl	%r9d, %r9d
	addq	%rax, %r9
	andq	%r8, %r9
.LBB19_88:
	pcmpeqb	%xmm7, %xmm1
	pmovmskb	%xmm1, %edi
	testl	%edi, %edi
	jne	.LBB19_91
	movl	$1, %edi
	jmp	.LBB19_90
.LBB19_43:
	xorl	%edi, %edi
.LBB19_90:
	addq	%rsi, %rax
	addq	$16, %rax
	addq	$16, %rsi
	jmp	.LBB19_37
	.p2align	4
.LBB19_19:
	leaq	(%rcx,%rsi,4), %rax
	movzwl	-2(%rax), %ecx
	movw	%cx, 72(%rsp)
	movq	-10(%rax), %rax
	movq	%rax, 64(%rsp)
	jmp	.LBB19_95
	.p2align	4
.LBB19_94:
	leaq	(,%r14,4), %rax
	addq	%r13, %rax
	movzwl	72(%rsp), %ecx
	movw	%cx, -2(%rax)
	movq	64(%rsp), %rcx
	movq	%rcx, -10(%rax)
.LBB19_95:
	movq	48(%rsp), %rcx
.LBB19_96:
	movq	120(%rsp), %r14
	incq	%r14
	movw	$-1, 560(%rsp)
	movq	$-1, 552(%rsp)
	movq	112(%rcx), %rcx
	xorl	%eax, %eax
	.p2align	4
.LBB19_97:
	xorl	%edx, %edx
	movq	%rcx, %r11
	subq	%rax, %r11
	seta	%r8b
	cmpq	$4, %rax
	ja	.LBB19_100
	movzbl	88(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_101
	xorl	%r10d, %r10d
	xorl	%esi, %esi
	.p2align	4
.LBB19_115:
	cmpq	%r11, %r10
	adcq	$0, %rsi
	cmpq	$4, %r10
	ja	.LBB19_124
	leaq	(%r10,%rax), %r9
	movzbl	64(%rsp,%r10,2), %edi
	cmpq	$4, %r9
	ja	.LBB19_121
	cmpb	$-1, %dil
	je	.LBB19_119
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	552(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 552(%rsp,%r9,2)
.LBB19_119:
	movzbl	65(%rsp,%r10,2), %edi
	cmpb	$-1, %dil
	je	.LBB19_113
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	553(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 553(%rsp,%r9,2)
	jmp	.LBB19_113
	.p2align	4
.LBB19_121:
	cmpb	$-1, %dil
	jne	.LBB19_123
	cmpb	$-1, 65(%rsp,%r10,2)
	jne	.LBB19_123
.LBB19_113:
	cmpq	%r11, %r10
	jae	.LBB19_101
	movq	%rsi, %r10
	cmpq	%r11, %rsi
	jbe	.LBB19_115
.LBB19_101:
	movzbl	89(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_110
	xorl	%r10d, %r10d
	xorl	%esi, %esi
	.p2align	4
.LBB19_103:
	cmpq	%r11, %r10
	adcq	$0, %rsi
	cmpq	$4, %r10
	ja	.LBB19_124
	leaq	(%r10,%rax), %r9
	movzbl	64(%rsp,%r10,2), %edi
	cmpq	$5, %r9
	jae	.LBB19_105
	cmpb	$-1, %dil
	je	.LBB19_108
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	553(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 553(%rsp,%r9,2)
	jmp	.LBB19_108
	.p2align	4
.LBB19_105:
	cmpb	$-1, %dil
	jne	.LBB19_123
.LBB19_108:
	cmpq	%r11, %r10
	jae	.LBB19_110
	movq	%rsi, %r10
	cmpq	%r11, %rsi
	jbe	.LBB19_103
.LBB19_110:
	cmpq	%rax, %rcx
	jbe	.LBB19_112
	movb	%r8b, %dl
	addq	%rdx, %rax
	cmpq	%rcx, %rax
	jbe	.LBB19_97
	jmp	.LBB19_112
.LBB19_91:
	movzbl	(%r13,%r9), %eax
	testb	%al, %al
	jns	.LBB19_92
.LBB19_93:
	movzwl	1066(%rsp), %r10d
	movw	%r10w, 560(%rsp)
	movq	1058(%rsp), %r10
	movq	%r10, 552(%rsp)
	movzwl	72(%rsp), %r10d
	leaq	562(%rsp), %rcx
	movw	%r10w, 8(%rcx)
	movq	64(%rsp), %r10
	movq	%r10, (%rcx)
	andb	$1, %al
	leaq	-16(%r9), %r10
	andq	%r8, %r10
	movb	%dl, (%r13,%r9)
	movb	%dl, 16(%r13,%r10)
	movq	48(%rsp), %rcx
	movdqu	16(%rcx), %xmm0
	movzbl	%al, %eax
	movd	%eax, %xmm1
	shufps	$228, %xmm6, %xmm1
	psubq	%xmm1, %xmm0
	movdqu	%xmm0, 16(%rcx)
	negq	%r9
	leaq	(%r9,%r9,4), %rax
	movl	568(%rsp), %edx
	movl	%edx, -4(%r13,%rax,4)
	movdqu	552(%rsp), %xmm0
	movdqu	%xmm0, -20(%r13,%rax,4)
	jmp	.LBB19_96
.LBB19_35:
	movl	$1, %edx
	movq	184(%rsp), %r8
	movb	$1, %r9b
	movq	%rax, %rsi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTThAhj9_EAAhj2_j5_EE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_BY_NtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rcx
	movq	%rsi, %rax
	jmp	.LBB19_36
.LBB19_92:
	movdqa	(%r13), %xmm0
	pmovmskb	%xmm0, %eax
	rep		bsfl	%eax, %r9d
	movzbl	(%r13,%r9), %eax
	jmp	.LBB19_93
.LBB19_130:
	addq	%r10, %rcx
	movzbl	-1(%rcx), %eax
.LBB19_131:
	movaps	1808(%rsp), %xmm6
	movaps	1824(%rsp), %xmm7
	.seh_startepilogue
	addq	$1848, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	.seh_endepilogue
	retq
.LBB19_58:
	leaq	anon.da267e598a0a058d3d272408d5046ada.139(%rip), %r8
	movl	$250, %edx
	movq	56(%rsp), %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_61:
	movq	%r10, %rcx
.LBB19_62:
	leaq	anon.da267e598a0a058d3d272408d5046ada.138(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_7:
	cmpq	$5, %rcx
	jae	.LBB19_8
	movzbl	89(%rsp,%rcx,2), %esi
	cmpq	$0, 104(%r10)
	jne	.LBB19_135
	movzbl	147(%r10), %r8d
	movups	anon.da267e598a0a058d3d272408d5046ada.23+11(%rip), %xmm0
	movups	%xmm0, 579(%rsp)
	movdqu	anon.da267e598a0a058d3d272408d5046ada.23(%rip), %xmm0
	movdqu	%xmm0, 568(%rsp)
	xorl	%eax, %eax
	xorl	%edx, %edx
	xorl	%r9d, %r9d
	.p2align	4
.LBB19_127:
	movzbl	568(%rsp,%r9), %r10d
	movl	%r10d, %ecx
	subb	$9, %cl
	cmpb	$9, %cl
	movl	$2, %ecx
	sbbl	$0, %ecx
	cmpb	$9, %r10b
	movl	$0, %r11d
	cmovaeq	%rcx, %r11
	cmpl	%r8d, %r11d
	jne	.LBB19_128
	incq	%r9
	cmpq	$27, %r9
	jne	.LBB19_127
	jmp	.LBB19_133
.LBB19_128:
	movzbl	%r10b, %ecx
	cmpb	$26, %r10b
	movq	80(%rsp), %r11
	ja	.LBB19_153
	movzbl	(%r11,%rcx), %ecx
	movl	%ecx, %r10d
	shrb	%r10b
	addb	%r10b, %al
	andb	$1, %cl
	addb	%cl, %dl
	cmpq	$26, %r9
	leaq	1(%r9), %r9
	jne	.LBB19_127
	jmp	.LBB19_134
.LBB19_133:
	movq	80(%rsp), %r11
.LBB19_134:
	xorl	%ecx, %ecx
	movb	$7, %r8b
	subb	%al, %r8b
	movzbl	%r8b, %eax
	cmovbl	%ecx, %eax
	movl	%eax, %r8d
	subb	%dl, %r8b
	movzbl	%r8b, %edx
	cmovbl	%ecx, %edx
	addb	%al, %dl
	movzbl	%dl, %eax
	movzbl	%sil, %esi
	cmpb	%sil, %al
	cmovbl	%eax, %esi
	movq	48(%rsp), %r10
.LBB19_135:
	movq	80(%r10), %rcx
	movq	88(%r10), %rdx
	movq	%r11, %r8
	movq	%r10, %rbx
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	%rax, %r10
	cmpq	$0, 64(%rbx)
	je	.LBB19_136
.LBB19_137:
	movq	48(%rbx), %rcx
	movq	56(%rbx), %r8
	movq	%r10, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	xorl	%edi, %edi
	pcmpeqd	%xmm1, %xmm1
	xorl	%r11d, %r11d
.LBB19_138:
	andq	%r8, %r10
	movdqu	(%rcx,%r10), %xmm2
	movdqa	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb	%xmm3, %ebx
	testl	%ebx, %ebx
	movl	%esi, %eax
	je	.LBB19_141
.LBB19_139:
	rep		bsfl	%ebx, %r15d
	addq	%r10, %r15
	andq	%r8, %r15
	movq	%r15, %r14
	negq	%r14
	leaq	(%r14,%r14,8), %r14
	leaq	(%r14,%r14,2), %r14
	subq	%r15, %r14
	movq	80(%rsp), %r15
	movdqu	(%r15), %xmm3
	movdqu	11(%r15), %xmm4
	movdqu	-28(%rcx,%r14), %xmm5
	pcmpeqb	%xmm3, %xmm5
	movdqu	-17(%rcx,%r14), %xmm3
	pcmpeqb	%xmm4, %xmm3
	pand	%xmm5, %xmm3
	pmovmskb	%xmm3, %ebp
	cmpl	$65535, %ebp
	je	.LBB19_151
	leal	-1(%rbx), %ebp
	andw	%bx, %bp
	movl	%ebp, %ebx
	jne	.LBB19_139
.LBB19_141:
	cmpq	$1, %rdi
	je	.LBB19_145
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	je	.LBB19_143
	rep		bsfl	%r9d, %r9d
	addq	%r10, %r9
	andq	%r8, %r9
.LBB19_145:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %edi
	testl	%edi, %edi
	jne	.LBB19_148
	movl	$1, %edi
	jmp	.LBB19_147
.LBB19_143:
	xorl	%edi, %edi
.LBB19_147:
	addq	%r11, %r10
	addq	$16, %r10
	addq	$16, %r11
	jmp	.LBB19_138
.LBB19_151:
	addq	%r14, %rcx
	movb	%al, -1(%rcx)
	jmp	.LBB19_131
.LBB19_148:
	movzbl	(%rcx,%r9), %r10d
	testb	%r10b, %r10b
	jns	.LBB19_149
.LBB19_150:
	andb	$1, %r10b
	leaq	-16(%r9), %r11
	andq	%r8, %r11
	movb	%dl, (%rcx,%r9)
	movb	%dl, 16(%rcx,%r11)
	movq	48(%rsp), %r8
	movdqu	64(%r8), %xmm0
	movzbl	%r10b, %edx
	movd	%edx, %xmm1
	movsd	%xmm1, %xmm6
	psubq	%xmm6, %xmm0
	movdqu	%xmm0, 64(%r8)
	movq	%r9, %rdx
	negq	%rdx
	leaq	(%rdx,%rdx,8), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	subq	%r9, %rdx
	movq	80(%rsp), %r8
	movdqu	(%r8), %xmm0
	movdqu	11(%r8), %xmm1
	movdqu	%xmm1, -17(%rcx,%rdx)
	movdqu	%xmm0, -28(%rcx,%rdx)
	movb	%al, -1(%rcx,%rdx)
	jmp	.LBB19_131
.LBB19_152:
	leaq	anon.da267e598a0a058d3d272408d5046ada.142(%rip), %r9
	movl	$250, %r8d
	xorl	%ecx, %ecx
	movq	%rbx, %rdx
	callq	_RNvNtNtCs9OCfJ4KTcUD_4core5slice5index16slice_index_fail
	ud2
.LBB19_124:
	leaq	anon.da267e598a0a058d3d272408d5046ada.21(%rip), %r8
	movl	$5, %edx
	movq	%r10, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_123:
	leaq	anon.da267e598a0a058d3d272408d5046ada.22(%rip), %r8
	movl	$5, %edx
	movq	%r9, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_70:
	leaq	anon.da267e598a0a058d3d272408d5046ada.137(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_100:
	leaq	anon.da267e598a0a058d3d272408d5046ada.20(%rip), %r8
	movl	$5, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_86:
	leaq	anon.da267e598a0a058d3d272408d5046ada.141(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_84:
	leaq	anon.da267e598a0a058d3d272408d5046ada.140(%rip), %r8
	movl	$250, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_136:
	leaq	48(%rbx), %rcx
	leaq	80(%rbx), %r8
	movl	$1, %edx
	movb	$1, %r9b
	movq	%r10, %rdi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTAhj1b_hEE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_hNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	%rdi, %r10
	jmp	.LBB19_137
.LBB19_149:
	movdqa	(%rcx), %xmm0
	pmovmskb	%xmm0, %r9d
	rep		bsfl	%r9d, %r9d
	movzbl	(%rcx,%r9), %r10d
	jmp	.LBB19_150
.LBB19_8:
	leaq	anon.da267e598a0a058d3d272408d5046ada.19(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_153:
	leaq	anon.da267e598a0a058d3d272408d5046ada.24(%rip), %r8
	movl	$27, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
	.seh_endproc

	.def	_RNvMs1_NtCs4DgSkW5ScxR_19sichuan_mahjong_sim6runnerNtB5_6Runner11first_legal;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,_RNvMs1_NtCs4DgSkW5ScxR_19sichuan_mahjong_sim6runnerNtB5_6Runner11first_legal,unique,20
	.globl	_RNvMs1_NtCs4DgSkW5ScxR_19sichuan_mahjong_sim6runnerNtB5_6Runner11first_legal
	.p2align	4
