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
	subq	$1864, %rsp
	.seh_stackalloc 1864
	movdqa	%xmm7, 1840(%rsp)
	.seh_savexmm %xmm7, 1840
	movaps	%xmm6, 1824(%rsp)
	.seh_savexmm %xmm6, 1824
	.seh_endprologue
	movq	%rdx, %r8
	movq	%rcx, %r11
	cmpq	$0, 72(%rcx)
	movq	%rdx, 88(%rsp)
	movq	%rcx, 48(%rsp)
	je	.LBB19_7
	movq	80(%r11), %rcx
	movq	88(%r11), %rdx
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %r11
	movq	%rax, %r8
	shrq	$57, %r8
	movq	48(%r11), %rcx
	movq	56(%r11), %rdx
	movd	%r8d, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	xorl	%esi, %esi
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
	movq	88(%rsp), %r8
	movdqu	(%r8), %xmm3
	movdqu	11(%r8), %xmm4
	movdqu	-28(%rcx,%r10), %xmm5
	pcmpeqb	%xmm3, %xmm5
	movdqu	-17(%rcx,%r10), %xmm3
	pcmpeqb	%xmm4, %xmm3
	pand	%xmm5, %xmm3
	pmovmskb	%xmm3, %r11d
	cmpl	$65535, %r11d
	je	.LBB19_127
	leal	-1(%r9), %r10d
	andw	%r9w, %r10w
	movl	%r10d, %r9d
	movq	48(%rsp), %r11
	jne	.LBB19_3
	.p2align	4
.LBB19_5:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	movq	88(%rsp), %r8
	jne	.LBB19_7
	addq	%rsi, %rax
	addq	$16, %rax
	addq	$16, %rsi
	jmp	.LBB19_2
.LBB19_7:
	movq	$-1, 105(%rsp)
	movb	$-1, 113(%rsp)
	movb	$0, 104(%rsp)
	movq	$3, 288(%rsp)
	movw	$256, 296(%rsp)
	movb	$2, 298(%rsp)
	leaq	120(%r11), %rax
	movq	%rax, 224(%rsp)
	pcmpeqd	%xmm7, %xmm7
	leaq	1074(%rsp), %r15
	leaq	32(%r11), %rax
	movq	%rax, 216(%rsp)
	movaps	__xmm@ffffffffffffffff0000000000000000(%rip), %xmm6
	xorl	%eax, %eax
	jmp	.LBB19_9
	.p2align	4
.LBB19_8:
	movzwl	576(%rsp), %eax
	movw	%ax, 112(%rsp)
	movq	568(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	240(%rsp), %rax
	cmpq	$3, %rax
	movq	88(%rsp), %r8
	movq	48(%rsp), %r11
	je	.LBB19_120
.LBB19_9:
	leaq	1(%rax), %rcx
	movq	%rcx, 240(%rsp)
	movq	%rcx, 280(%rsp)
	movzbl	296(%rsp,%rax), %eax
	movb	%al, 47(%rsp)
	movzbl	%al, %eax
	leaq	(%rax,%rax,8), %r9
	movzbl	8(%r8,%r9), %ecx
	movb	%cl, 144(%rsp)
	movq	(%r8,%r9), %rcx
	movq	%rcx, 136(%rsp)
	movzbl	8(%r8,%r9), %ecx
	leaq	151(%rsp), %rdx
	movb	%cl, 8(%rdx)
	movq	%r9, %r12
	movq	(%r8,%r9), %rcx
	movq	%rcx, (%rdx)
	movb	%al, 150(%rsp)
	cmpq	$0, 24(%r11)
	je	.LBB19_17
	movq	32(%r11), %rcx
	movq	40(%r11), %rdx
	leaq	150(%rsp), %r8
	movq	%r11, %rsi
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRThAhj9_EECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	%rax, %r8
	shrq	$57, %r8
	movq	(%rsi), %rcx
	movq	8(%rsi), %rdx
	movd	%r8d, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	movzbl	159(%rsp), %r14d
	movq	151(%rsp), %r9
	xorl	%r10d, %r10d
.LBB19_11:
	andq	%rdx, %rax
	movdqu	(%rcx,%rax), %xmm1
	movdqa	%xmm1, %xmm2
	pcmpeqb	%xmm0, %xmm2
	pmovmskb	%xmm2, %r11d
	testl	%r11d, %r11d
	je	.LBB19_15
.LBB19_12:
	rep		bsfl	%r11d, %esi
	addq	%rax, %rsi
	andq	%rdx, %rsi
	negq	%rsi
	leaq	(%rsi,%rsi,4), %rsi
	movzbl	47(%rsp), %r8d
	cmpb	-20(%rcx,%rsi,4), %r8b
	jne	.LBB19_14
	movzbl	-11(%rcx,%rsi,4), %edi
	movq	-19(%rcx,%rsi,4), %rbx
	xorq	%r9, %rbx
	xorl	%r14d, %edi
	movzbl	%dil, %edi
	orq	%rbx, %rdi
	je	.LBB19_88
.LBB19_14:
	leal	-1(%r11), %esi
	andw	%r11w, %si
	movl	%esi, %r11d
	jne	.LBB19_12
	.p2align	4
.LBB19_15:
	pcmpeqb	%xmm7, %xmm1
	pmovmskb	%xmm1, %r11d
	testl	%r11d, %r11d
	movq	88(%rsp), %r8
	movq	48(%rsp), %r11
	jne	.LBB19_17
	addq	%r10, %rax
	addq	$16, %rax
	addq	$16, %r10
	jmp	.LBB19_11
	.p2align	4
.LBB19_17:
	leaq	(%r8,%r12), %rax
	movq	%rax, 232(%rsp)
	addq	224(%rsp), %r12
	movq	%r12, 248(%rsp)
	movq	112(%r11), %rdi
	leaq	1575(%rsp), %rax
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
	leaq	1076(%rsp), %rcx
	xorl	%edx, %edx
	callq	memset
	movb	$0, 1574(%rsp)
	movw	$0, 1074(%rsp)
	movl	$1, %r14d
	xorl	%eax, %eax
	movq	%rdi, 120(%rsp)
	jmp	.LBB19_19
	.p2align	4
.LBB19_18:
	movl	$250, %r8d
	leaq	1574(%rsp), %rcx
	leaq	304(%rsp), %rdx
	callq	memcpy
	movl	$500, %r8d
	movq	%r15, %rcx
	leaq	568(%rsp), %rdx
	callq	memcpy
	movq	%r12, %rax
	testq	%r14, %r14
	je	.LBB19_70
.LBB19_19:
	cmpq	$9, %rax
	je	.LBB19_70
	movq	%rax, 96(%rsp)
	movdqu	%xmm7, 538(%rsp)
	movdqa	%xmm7, 528(%rsp)
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
	movl	$500, %r8d
	leaq	568(%rsp), %rcx
	xorl	%edx, %edx
	callq	memset
	cmpq	$251, %r14
	jae	.LBB19_146
	movq	96(%rsp), %rcx
	leaq	1(%rcx), %r12
	movq	248(%rsp), %rax
	movzbl	(%rax,%rcx), %eax
	movb	%al, 63(%rsp)
	movzbl	%al, %edx
	leaq	(%rsp,%r14,2), %rax
	addq	$1074, %rax
	movq	%rax, 272(%rsp)
	xorl	%r14d, %r14d
	movq	%r15, %r11
	.p2align	4
.LBB19_22:
	cmpq	272(%rsp), %r11
	je	.LBB19_18
	movzwl	(%r11), %ecx
	cmpq	$250, %rcx
	jae	.LBB19_152
	addq	$2, %r11
	imull	$205, %ecx, %eax
	movzwl	%ax, %eax
	shrl	$11, %eax
	imull	$52, %eax, %ebx
	movzbl	%bh, %esi
	leal	(%rsi,%rsi,4), %r8d
	subb	%r8b, %al
	leal	(%rcx,%rcx,4), %r8d
	leal	(%rcx,%r8,8), %r8d
	movzwl	%r8w, %r8d
	shrl	$11, %r8d
	addb	%al, %r8b
	cmpb	63(%rsp), %r8b
	ja	.LBB19_22
	movq	%r12, 128(%rsp)
	movq	%r14, 64(%rsp)
	movzwl	%cx, %r9d
	shrl	%r9d
	imull	$205, %r9d, %r10d
	movzwl	%r10w, %r10d
	shrl	$10, %r10d
	leal	(%r10,%r10,4), %r10d
	subb	%r10b, %r9b
	movzbl	%r9b, %r10d
	movq	96(%rsp), %r9
	movzbl	%r8b, %esi
	movq	%rdx, %rbx
	movq	%rsi, 200(%rsp)
	subq	%rsi, %rbx
	movq	%rdi, %rsi
	movq	%r10, 208(%rsp)
	subq	%r10, %rsi
	cmpq	%rsi, %rdi
	movq	%rsi, 184(%rsp)
	cmovbq	%rdi, %rsi
	cmpq	%rsi, %rbx
	movq	%rbx, 192(%rsp)
	cmovbq	%rbx, %rsi
	cmpq	$7, %r9
	movzbl	1574(%rsp,%rcx), %r12d
	movl	$0, %r8d
	cmovaeq	%r8, %rsi
	movzbl	%al, %eax
	leal	(%rax,%rax,4), %eax
	movzbl	%al, %eax
	movq	%rax, 176(%rsp)
	testb	$1, %cl
	movq	%r11, 168(%rsp)
	movq	%rsi, 160(%rsp)
	jne	.LBB19_55
	xorl	%r8d, %r8d
	xorl	%eax, %eax
	.p2align	4
.LBB19_27:
	cmpq	%rsi, %r8
	adcq	$0, %rax
	movq	%rax, 256(%rsp)
	movq	192(%rsp), %r14
	subq	%r8, %r14
	movq	184(%rsp), %rbp
	subq	%r8, %rbp
	movq	200(%rsp), %rax
	leaq	(%r8,%rax), %r11
	movq	176(%rsp), %rax
	addq	%r8, %rax
	leaq	(%rax,%rax,4), %rax
	movq	208(%rsp), %rcx
	movq	%r8, 264(%rsp)
	addq	%r8, %rcx
	addq	%rcx, %rax
	xorl	%r13d, %r13d
	xorl	%esi, %esi
	.p2align	4
.LBB19_28:
	cmpq	%rbp, %r13
	adcq	$0, %rsi
	leaq	(,%r13,2), %rcx
	addq	%r13, %rcx
	cmpq	%r14, %rcx
	ja	.LBB19_53
	leaq	(%r11,%rcx), %rbx
	leaq	(%rax,%r13), %r10
	addq	%r10, %r10
	leaq	(%r11,%rcx), %r15
	addq	$2, %r15
	cmpq	%rdx, %r15
	jbe	.LBB19_37
	cmpq	%rdx, %rbx
	ja	.LBB19_50
	subb	136(%rsp,%r9), %bl
	movzbl	%bl, %ecx
	movl	$0, %r8d
	cmovbl	%r8d, %ecx
	addb	%r12b, %cl
	movzbl	%cl, %ecx
	movl	$255, %r8d
	cmovbl	%r8d, %ecx
	cmpq	$249, %r10
	ja	.LBB19_150
	movzbl	304(%rsp,%r10), %r8d
	cmpb	%r8b, %cl
	jae	.LBB19_50
	cmpb	$-1, %r8b
	jne	.LBB19_36
	movq	64(%rsp), %r8
	cmpq	$249, %r8
	ja	.LBB19_149
	movw	%r10w, 568(%rsp,%r8,2)
	incq	%r8
	movq	%r8, 64(%rsp)
.LBB19_36:
	movb	%cl, 304(%rsp,%r10)
	cmpq	%rbp, %r13
	jb	.LBB19_51
	jmp	.LBB19_53
	.p2align	4
.LBB19_37:
	xorl	%ecx, %ecx
	cmpq	%rdx, %rbx
	seta	%cl
	movl	%ebx, %r8d
	cmoval	%r15d, %r8d
	movzbl	136(%rsp,%r9), %r9d
	subb	%r9b, %r8b
	movzbl	%r8b, %r8d
	movl	$0, %edi
	cmovbl	%edi, %r8d
	addb	%r12b, %r8b
	movzbl	%r8b, %edi
	movl	$255, %r8d
	cmovbl	%r8d, %edi
	orq	%r10, %rcx
	cmpq	$249, %r10
	ja	.LBB19_151
	movzbl	304(%rsp,%rcx), %r8d
	cmpb	%r8b, %dil
	jae	.LBB19_43
	cmpb	$-1, %r8b
	jne	.LBB19_42
	movq	64(%rsp), %r8
	cmpq	$249, %r8
	ja	.LBB19_149
	movw	%cx, 568(%rsp,%r8,2)
	incq	%r8
	movq	%r8, 64(%rsp)
.LBB19_42:
	movb	%dil, 304(%rsp,%rcx)
.LBB19_43:
	cmpq	%rdx, %rbx
	ja	.LBB19_44
	subb	%r9b, %r15b
	movzbl	%r15b, %ecx
	movl	$0, %r8d
	cmovbl	%r8d, %ecx
	addb	%r12b, %cl
	movzbl	%cl, %ecx
	movl	$255, %r8d
	cmovbl	%r8d, %ecx
	movzbl	305(%rsp,%r10), %r9d
	cmpb	%r9b, %cl
	jae	.LBB19_44
	orq	$1, %r10
	cmpb	$-1, %r9b
	jne	.LBB19_49
	movq	64(%rsp), %r8
	cmpq	$249, %r8
	ja	.LBB19_149
	movw	%r10w, 568(%rsp,%r8,2)
	incq	%r8
	movq	%r8, 64(%rsp)
.LBB19_49:
	movb	%cl, 304(%rsp,%r10)
	movq	96(%rsp), %r9
	.p2align	4
.LBB19_50:
	cmpq	%rbp, %r13
	jb	.LBB19_51
	jmp	.LBB19_53
	.p2align	4
.LBB19_44:
	movq	96(%rsp), %r9
	cmpq	%rbp, %r13
	jae	.LBB19_53
.LBB19_51:
	movq	%rsi, %r13
	cmpq	%rbp, %rsi
	jbe	.LBB19_28
.LBB19_53:
	movq	160(%rsp), %rsi
	cmpq	%rsi, 264(%rsp)
	movq	168(%rsp), %r11
	jae	.LBB19_69
	movq	256(%rsp), %rax
	movq	%rax, %r8
	cmpq	%rsi, %rax
	jbe	.LBB19_27
	jmp	.LBB19_69
.LBB19_55:
	xorl	%r8d, %r8d
	xorl	%eax, %eax
	.p2align	4
.LBB19_56:
	cmpq	%rsi, %r8
	adcq	$0, %rax
	movq	192(%rsp), %r10
	subq	%r8, %r10
	movq	184(%rsp), %r11
	subq	%r8, %r11
	movq	200(%rsp), %rcx
	leaq	(%r8,%rcx), %rsi
	movq	176(%rsp), %rcx
	addq	%r8, %rcx
	leaq	(%rcx,%rcx,4), %r14
	movq	208(%rsp), %rcx
	addq	%r8, %rcx
	addq	%rcx, %r14
	xorl	%ebp, %ebp
	xorl	%r13d, %r13d
	.p2align	4
.LBB19_57:
	cmpq	%r11, %rbp
	adcq	$0, %r13
	leaq	(,%rbp,2), %r9
	addq	%rbp, %r9
	cmpq	%r10, %r9
	ja	.LBB19_67
	addq	%rsi, %r9
	cmpq	%rdx, %r9
	ja	.LBB19_65
	leaq	(%r14,%rbp), %rdi
	leaq	1(,%rdi,2), %rcx
	addq	%rdi, %rdi
	movq	96(%rsp), %rbx
	subb	136(%rsp,%rbx), %r9b
	movzbl	%r9b, %r9d
	movl	$0, %ebx
	cmovbl	%ebx, %r9d
	addb	%r12b, %r9b
	movzbl	%r9b, %r9d
	movl	$255, %ebx
	cmovbl	%ebx, %r9d
	cmpq	$249, %rdi
	ja	.LBB19_151
	movzbl	304(%rsp,%rcx), %ebx
	cmpb	%bl, %r9b
	jae	.LBB19_65
	cmpb	$-1, %bl
	jne	.LBB19_64
	movq	64(%rsp), %rdi
	cmpq	$249, %rdi
	ja	.LBB19_149
	movw	%cx, 568(%rsp,%rdi,2)
	incq	%rdi
	movq	%rdi, 64(%rsp)
.LBB19_64:
	movb	%r9b, 304(%rsp,%rcx)
.LBB19_65:
	cmpq	%r11, %rbp
	jae	.LBB19_67
	movq	%r13, %rbp
	cmpq	%r11, %r13
	jbe	.LBB19_57
.LBB19_67:
	movq	160(%rsp), %rsi
	cmpq	%rsi, %r8
	movq	168(%rsp), %r11
	jae	.LBB19_69
	movq	%rax, %r8
	cmpq	%rsi, %rax
	jbe	.LBB19_56
.LBB19_69:
	leaq	1074(%rsp), %r15
	movq	120(%rsp), %rdi
	movq	64(%rsp), %r14
	movq	128(%rsp), %r12
	jmp	.LBB19_22
	.p2align	4
.LBB19_70:
	movw	$-1, 80(%rsp)
	movq	$-1, 72(%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	48(%rsp), %r8
	.p2align	4
.LBB19_71:
	cmpq	%rdi, %rcx
	adcq	$0, %rdx
	leaq	(%rcx,%rcx), %rax
	cmpq	$249, %rax
	ja	.LBB19_155
	cmpq	$4, %rcx
	ja	.LBB19_154
	movzwl	1574(%rsp,%rcx,2), %eax
	movw	%ax, 72(%rsp,%rcx,2)
	cmpq	%rdi, %rcx
	jae	.LBB19_75
	movq	%rdx, %rcx
	cmpq	%rdi, %rdx
	jbe	.LBB19_71
.LBB19_75:
	movq	232(%rsp), %rdx
	movzbl	8(%rdx), %eax
	leaq	1076(%rsp), %rcx
	movb	%al, 7(%rcx)
	movq	(%rdx), %rax
	movq	%rax, -1(%rcx)
	movzbl	47(%rsp), %eax
	movb	%al, 1074(%rsp)
	movq	32(%r8), %rcx
	movq	40(%r8), %rdx
	movq	%r15, %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRThAhj9_EECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rcx
	cmpq	$0, 16(%rcx)
	je	.LBB19_118
.LBB19_76:
	movq	(%rcx), %r13
	movq	8(%rcx), %r8
	movq	%rax, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	movzbl	1083(%rsp), %r10d
	movq	1075(%rsp), %r11
	xorl	%edi, %edi
	xorl	%esi, %esi
.LBB19_77:
	andq	%r8, %rax
	movdqu	(%r13,%rax), %xmm1
	movdqa	%xmm1, %xmm2
	pcmpeqb	%xmm0, %xmm2
	pmovmskb	%xmm2, %ebx
	testl	%ebx, %ebx
	je	.LBB19_81
.LBB19_78:
	rep		bsfl	%ebx, %r14d
	addq	%rax, %r14
	andq	%r8, %r14
	negq	%r14
	leaq	(%r14,%r14,4), %r14
	movzbl	47(%rsp), %ecx
	cmpb	-20(%r13,%r14,4), %cl
	jne	.LBB19_80
	movzbl	-11(%r13,%r14,4), %r15d
	movq	-19(%r13,%r14,4), %r12
	xorq	%r11, %r12
	xorl	%r10d, %r15d
	movzbl	%r15b, %r15d
	orq	%r12, %r15
	leaq	1074(%rsp), %r15
	je	.LBB19_89
.LBB19_80:
	leal	-1(%rbx), %ebp
	andw	%bx, %bp
	movl	%ebp, %ebx
	jne	.LBB19_78
	.p2align	4
.LBB19_81:
	cmpq	$1, %rdi
	je	.LBB19_84
	pmovmskb	%xmm1, %r9d
	testl	%r9d, %r9d
	je	.LBB19_86
	rep		bsfl	%r9d, %r9d
	addq	%rax, %r9
	andq	%r8, %r9
.LBB19_84:
	pcmpeqb	%xmm7, %xmm1
	pmovmskb	%xmm1, %edi
	testl	%edi, %edi
	jne	.LBB19_91
	movl	$1, %edi
	jmp	.LBB19_87
.LBB19_86:
	xorl	%edi, %edi
.LBB19_87:
	addq	%rsi, %rax
	addq	$16, %rax
	addq	$16, %rsi
	jmp	.LBB19_77
	.p2align	4
.LBB19_88:
	leaq	(%rcx,%rsi,4), %rax
	movzwl	-2(%rax), %ecx
	movw	%cx, 80(%rsp)
	movq	-10(%rax), %rax
	movq	%rax, 72(%rsp)
	jmp	.LBB19_90
	.p2align	4
.LBB19_89:
	leaq	(,%r14,4), %rax
	addq	%r13, %rax
	movzwl	80(%rsp), %ecx
	movw	%cx, -2(%rax)
	movq	72(%rsp), %rcx
	movq	%rcx, -10(%rax)
.LBB19_90:
	movq	48(%rsp), %rcx
	jmp	.LBB19_93
	.p2align	4
.LBB19_91:
	movzbl	(%r13,%r9), %eax
	testb	%al, %al
	jns	.LBB19_119
.LBB19_92:
	movzwl	1082(%rsp), %r10d
	movw	%r10w, 576(%rsp)
	movq	1074(%rsp), %r10
	movq	%r10, 568(%rsp)
	movzwl	80(%rsp), %r10d
	leaq	578(%rsp), %rcx
	movw	%r10w, 8(%rcx)
	movq	72(%rsp), %r10
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
	movl	584(%rsp), %edx
	movl	%edx, -4(%r13,%rax,4)
	movdqu	568(%rsp), %xmm0
	movdqu	%xmm0, -20(%r13,%rax,4)
.LBB19_93:
	movw	$-1, 576(%rsp)
	movq	$-1, 568(%rsp)
	movq	112(%rcx), %rcx
	xorl	%eax, %eax
	.p2align	4
.LBB19_94:
	xorl	%edx, %edx
	movq	%rcx, %r11
	subq	%rax, %r11
	seta	%r8b
	cmpq	$4, %rax
	ja	.LBB19_153
	movzbl	104(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_107
	xorl	%r10d, %r10d
	xorl	%esi, %esi
	.p2align	4
.LBB19_97:
	cmpq	%r11, %r10
	adcq	$0, %rsi
	cmpq	$4, %r10
	ja	.LBB19_147
	leaq	(%r10,%rax), %r9
	movzbl	72(%rsp,%r10,2), %edi
	cmpq	$4, %r9
	ja	.LBB19_103
	cmpb	$-1, %dil
	je	.LBB19_101
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	568(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 568(%rsp,%r9,2)
.LBB19_101:
	movzbl	73(%rsp,%r10,2), %edi
	cmpb	$-1, %dil
	je	.LBB19_105
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	569(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 569(%rsp,%r9,2)
	jmp	.LBB19_105
	.p2align	4
.LBB19_103:
	cmpb	$-1, %dil
	jne	.LBB19_148
	cmpb	$-1, 73(%rsp,%r10,2)
	jne	.LBB19_148
.LBB19_105:
	cmpq	%r11, %r10
	jae	.LBB19_107
	movq	%rsi, %r10
	cmpq	%r11, %rsi
	jbe	.LBB19_97
.LBB19_107:
	movzbl	105(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_116
	xorl	%r10d, %r10d
	xorl	%esi, %esi
	.p2align	4
.LBB19_109:
	cmpq	%r11, %r10
	adcq	$0, %rsi
	cmpq	$4, %r10
	ja	.LBB19_147
	leaq	(%r10,%rax), %r9
	movzbl	72(%rsp,%r10,2), %edi
	cmpq	$5, %r9
	jae	.LBB19_113
	cmpb	$-1, %dil
	je	.LBB19_114
	addb	%bl, %dil
	movzbl	%dil, %edi
	movl	$255, %ebp
	cmovbl	%ebp, %edi
	movzbl	569(%rsp,%r9,2), %ebp
	cmpb	%bpl, %dil
	cmovael	%ebp, %edi
	movb	%dil, 569(%rsp,%r9,2)
	jmp	.LBB19_114
	.p2align	4
.LBB19_113:
	cmpb	$-1, %dil
	jne	.LBB19_148
.LBB19_114:
	cmpq	%r11, %r10
	jae	.LBB19_116
	movq	%rsi, %r10
	cmpq	%r11, %rsi
	jbe	.LBB19_109
.LBB19_116:
	cmpq	%rax, %rcx
	jbe	.LBB19_8
	movb	%r8b, %dl
	addq	%rdx, %rax
	cmpq	%rcx, %rax
	jbe	.LBB19_94
	jmp	.LBB19_8
.LBB19_118:
	movl	$1, %edx
	movq	216(%rsp), %r8
	movb	$1, %r9b
	movq	%rax, %rsi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTThAhj9_EAAhj2_j5_EE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_BY_NtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rcx
	movq	%rsi, %rax
	jmp	.LBB19_76
.LBB19_119:
	movdqa	(%r13), %xmm0
	pmovmskb	%xmm0, %eax
	rep		bsfl	%eax, %r9d
	movzbl	(%r13,%r9), %eax
	jmp	.LBB19_92
.LBB19_120:
	cmpq	$5, %rcx
	jae	.LBB19_158
	movzbl	105(%rsp,%rcx,2), %ebx
	cmpq	$0, 104(%r11)
	jne	.LBB19_130
	movzbl	147(%r11), %esi
	movups	anon.da267e598a0a058d3d272408d5046ada.23+11(%rip), %xmm0
	movups	%xmm0, 595(%rsp)
	movdqu	anon.da267e598a0a058d3d272408d5046ada.23(%rip), %xmm0
	movdqu	%xmm0, 584(%rsp)
	xorl	%eax, %eax
	xorl	%edx, %edx
	xorl	%r9d, %r9d
	movq	88(%rsp), %r8
	.p2align	4
.LBB19_123:
	movzbl	584(%rsp,%r9), %r10d
	movl	%r10d, %ecx
	subb	$9, %cl
	cmpb	$9, %cl
	movl	$2, %ecx
	sbbl	$0, %ecx
	cmpb	$9, %r10b
	movl	$0, %r11d
	cmovaeq	%rcx, %r11
	cmpl	%esi, %r11d
	jne	.LBB19_125
	incq	%r9
	cmpq	$27, %r9
	jne	.LBB19_123
	jmp	.LBB19_128
.LBB19_125:
	movzbl	%r10b, %ecx
	cmpb	$26, %r10b
	movq	48(%rsp), %r11
	ja	.LBB19_159
	movzbl	(%r8,%rcx), %ecx
	movl	%ecx, %r10d
	shrb	%r10b
	addb	%r10b, %al
	andb	$1, %cl
	addb	%cl, %dl
	cmpq	$26, %r9
	leaq	1(%r9), %r9
	jne	.LBB19_123
	jmp	.LBB19_129
.LBB19_127:
	addq	%r10, %rcx
	movzbl	-1(%rcx), %ebx
	jmp	.LBB19_145
.LBB19_128:
	movq	88(%rsp), %r8
	movq	48(%rsp), %r11
.LBB19_129:
	xorl	%ecx, %ecx
	movb	$7, %r9b
	subb	%al, %r9b
	movzbl	%r9b, %eax
	cmovbl	%ecx, %eax
	movl	%eax, %r9d
	subb	%dl, %r9b
	movzbl	%r9b, %edx
	cmovbl	%ecx, %edx
	addb	%al, %dl
	movzbl	%dl, %eax
	movzbl	%bl, %ebx
	cmpb	%bl, %al
	cmovbl	%eax, %ebx
.LBB19_130:
	movq	80(%r11), %rcx
	movq	88(%r11), %rdx
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rdx
	cmpq	$0, 64(%rdx)
	je	.LBB19_156
.LBB19_131:
	movq	48(%rdx), %rcx
	movq	56(%rdx), %r8
	movq	%rax, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	xorl	%r11d, %r11d
	pcmpeqd	%xmm1, %xmm1
	xorl	%r10d, %r10d
.LBB19_132:
	andq	%r8, %rax
	movdqu	(%rcx,%rax), %xmm2
	movdqa	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb	%xmm3, %esi
	testl	%esi, %esi
	je	.LBB19_135
.LBB19_133:
	rep		bsfl	%esi, %r14d
	addq	%rax, %r14
	andq	%r8, %r14
	movq	%r14, %rdi
	negq	%rdi
	leaq	(%rdi,%rdi,8), %rdi
	leaq	(%rdi,%rdi,2), %rdi
	subq	%r14, %rdi
	movq	88(%rsp), %r14
	movdqu	(%r14), %xmm3
	movdqu	11(%r14), %xmm4
	movdqu	-28(%rcx,%rdi), %xmm5
	pcmpeqb	%xmm3, %xmm5
	movdqu	-17(%rcx,%rdi), %xmm3
	pcmpeqb	%xmm4, %xmm3
	pand	%xmm5, %xmm3
	pmovmskb	%xmm3, %ebp
	cmpl	$65535, %ebp
	je	.LBB19_142
	leal	-1(%rsi), %edi
	andw	%si, %di
	movl	%edi, %esi
	jne	.LBB19_133
.LBB19_135:
	cmpq	$1, %r11
	je	.LBB19_138
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	je	.LBB19_140
	rep		bsfl	%r9d, %r9d
	addq	%rax, %r9
	andq	%r8, %r9
.LBB19_138:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %r11d
	testl	%r11d, %r11d
	jne	.LBB19_143
	movl	$1, %r11d
	jmp	.LBB19_141
.LBB19_140:
	xorl	%r11d, %r11d
.LBB19_141:
	addq	%r10, %rax
	addq	$16, %rax
	addq	$16, %r10
	jmp	.LBB19_132
.LBB19_142:
	addq	%rdi, %rcx
	movb	%bl, -1(%rcx)
	jmp	.LBB19_145
.LBB19_143:
	movzbl	(%rcx,%r9), %eax
	testb	%al, %al
	jns	.LBB19_157
.LBB19_144:
	andb	$1, %al
	leaq	-16(%r9), %r10
	andq	%r8, %r10
	movb	%dl, (%rcx,%r9)
	movb	%dl, 16(%rcx,%r10)
	movq	48(%rsp), %rdx
	movdqu	64(%rdx), %xmm0
	movzbl	%al, %eax
	movd	%eax, %xmm1
	movsd	%xmm1, %xmm6
	psubq	%xmm6, %xmm0
	movdqu	%xmm0, 64(%rdx)
	movq	%r9, %rax
	negq	%rax
	leaq	(%rax,%rax,8), %rax
	leaq	(%rax,%rax,2), %rax
	subq	%r9, %rax
	movq	88(%rsp), %rdx
	movdqu	(%rdx), %xmm0
	movdqu	11(%rdx), %xmm1
	movdqu	%xmm1, -17(%rcx,%rax)
	movdqu	%xmm0, -28(%rcx,%rax)
	movb	%bl, -1(%rcx,%rax)
.LBB19_145:
	movl	%ebx, %eax
	movaps	1824(%rsp), %xmm6
	movaps	1840(%rsp), %xmm7
	.seh_startepilogue
	addq	$1864, %rsp
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
.LBB19_146:
	leaq	anon.da267e598a0a058d3d272408d5046ada.142(%rip), %r9
	movl	$250, %r8d
	xorl	%ecx, %ecx
	movq	%r14, %rdx
	callq	_RNvNtNtCs9OCfJ4KTcUD_4core5slice5index16slice_index_fail
	ud2
.LBB19_147:
	leaq	anon.da267e598a0a058d3d272408d5046ada.21(%rip), %r8
	movl	$5, %edx
	movq	%r10, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_148:
	leaq	anon.da267e598a0a058d3d272408d5046ada.22(%rip), %r8
	movl	$5, %edx
	movq	%r9, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_149:
	leaq	anon.da267e598a0a058d3d272408d5046ada.139(%rip), %r8
	movl	$250, %edx
	movq	64(%rsp), %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_150:
	movq	%r10, %rcx
.LBB19_151:
	leaq	anon.da267e598a0a058d3d272408d5046ada.138(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_152:
	leaq	anon.da267e598a0a058d3d272408d5046ada.137(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_153:
	leaq	anon.da267e598a0a058d3d272408d5046ada.20(%rip), %r8
	movl	$5, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_154:
	leaq	anon.da267e598a0a058d3d272408d5046ada.141(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_155:
	leaq	anon.da267e598a0a058d3d272408d5046ada.140(%rip), %r8
	movl	$250, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_156:
	leaq	48(%rdx), %rcx
	leaq	80(%rdx), %r8
	movl	$1, %edx
	movb	$1, %r9b
	movq	%rax, %rsi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTAhj1b_hEE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_hNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	48(%rsp), %rdx
	movq	%rsi, %rax
	jmp	.LBB19_131
.LBB19_157:
	movdqa	(%rcx), %xmm0
	pmovmskb	%xmm0, %eax
	rep		bsfl	%eax, %r9d
	movzbl	(%rcx,%r9), %eax
	jmp	.LBB19_144
.LBB19_158:
	leaq	anon.da267e598a0a058d3d272408d5046ada.19(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_159:
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
