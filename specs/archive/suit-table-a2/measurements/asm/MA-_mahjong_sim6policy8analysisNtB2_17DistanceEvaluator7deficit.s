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
	subq	$1880, %rsp
	.seh_stackalloc 1880
	movdqa	%xmm7, 1856(%rsp)
	.seh_savexmm %xmm7, 1856
	movaps	%xmm6, 1840(%rsp)
	.seh_savexmm %xmm6, 1840
	.seh_endprologue
	movq	%rdx, %r15
	movq	%rcx, %r11
	cmpq	$0, 72(%rcx)
	movq	%rcx, 40(%rsp)
	je	.LBB19_10
	movq	80(%r11), %rcx
	movq	88(%r11), %rdx
	movq	%r15, %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	40(%rsp), %r11
	movq	%rax, %r8
	shrq	$57, %r8
	movq	48(%r11), %rcx
	movq	56(%r11), %rdx
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
	movdqu	(%r15), %xmm3
	movdqu	11(%r15), %xmm4
	movdqu	-28(%rcx,%r10), %xmm5
	pcmpeqb	%xmm3, %xmm5
	movdqu	-17(%rcx,%r10), %xmm3
	pcmpeqb	%xmm4, %xmm3
	pand	%xmm5, %xmm3
	pmovmskb	%xmm3, %r11d
	cmpl	$65535, %r11d
	je	.LBB19_108
	leal	-1(%r9), %r10d
	andw	%r9w, %r10w
	movl	%r10d, %r9d
	movq	40(%rsp), %r11
	jne	.LBB19_3
	.p2align	4
.LBB19_5:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	jne	.LBB19_10
	addq	%r8, %rax
	addq	$16, %rax
	addq	$16, %r8
	jmp	.LBB19_2
.LBB19_10:
	movq	$-1, 65(%rsp)
	movb	$-1, 73(%rsp)
	movb	$0, 64(%rsp)
	movq	$3, 304(%rsp)
	movw	$256, 312(%rsp)
	movb	$2, 314(%rsp)
	leaq	120(%r11), %rax
	movq	%rax, 168(%rsp)
	pcmpeqd	%xmm7, %xmm7
	leaq	1090(%rsp), %r12
	leaq	32(%r11), %rax
	movq	%rax, 160(%rsp)
	movaps	__xmm@ffffffffffffffff0000000000000000(%rip), %xmm6
	leaq	584(%rsp), %r14
	xorl	%eax, %eax
	movq	%r15, 80(%rsp)
	jmp	.LBB19_11
	.p2align	4
.LBB19_91:
	movzwl	592(%rsp), %eax
	movw	%ax, 72(%rsp)
	movq	584(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	184(%rsp), %rax
	cmpq	$3, %rax
	movq	40(%rsp), %r11
	je	.LBB19_7
.LBB19_11:
	leaq	1(%rax), %rcx
	movq	%rcx, 184(%rsp)
	movq	%rcx, 296(%rsp)
	movzbl	312(%rsp,%rax), %eax
	movb	%al, 38(%rsp)
	movzbl	%al, %eax
	leaq	(%rax,%rax,8), %r8
	movzbl	8(%r15,%r8), %ecx
	movb	%cl, 152(%rsp)
	movq	(%r15,%r8), %rcx
	movq	%rcx, 144(%rsp)
	movzbl	8(%r15,%r8), %ecx
	leaq	95(%rsp), %rdx
	movb	%cl, 8(%rdx)
	movq	%r8, %r13
	movq	(%r15,%r8), %rcx
	movq	%rcx, (%rdx)
	movb	%al, 94(%rsp)
	cmpq	$0, 24(%r11)
	je	.LBB19_20
	movq	32(%r11), %rcx
	movq	40(%r11), %rdx
	leaq	94(%rsp), %r8
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
	movzbl	103(%rsp), %r8d
	movq	95(%rsp), %r9
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
	leaq	(%rsi,%rsi,4), %rdi
	movzbl	38(%rsp), %ebx
	cmpb	-20(%rcx,%rdi,4), %bl
	jne	.LBB19_16
	movzbl	-11(%rcx,%rdi,4), %esi
	movq	-19(%rcx,%rdi,4), %rbx
	xorq	%r9, %rbx
	xorl	%r8d, %esi
	movzbl	%sil, %esi
	orq	%rbx, %rsi
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
	movq	40(%rsp), %r11
	jne	.LBB19_20
	addq	%r10, %rax
	addq	$16, %rax
	addq	$16, %r10
	jmp	.LBB19_13
	.p2align	4
.LBB19_20:
	leaq	(%r15,%r13), %rax
	movq	%rax, 176(%rsp)
	addq	168(%rsp), %r13
	movq	%r13, 192(%rsp)
	movq	112(%r11), %rsi
	leaq	1591(%rsp), %rax
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
	leaq	1092(%rsp), %rcx
	xorl	%edx, %edx
	callq	memset
	movb	$0, 1590(%rsp)
	movw	$0, 1090(%rsp)
	movl	$1, %r15d
	xorl	%edi, %edi
	movq	%rsi, 104(%rsp)
	jmp	.LBB19_21
	.p2align	4
.LBB19_29:
	movl	$250, %r8d
	leaq	1590(%rsp), %rcx
	leaq	320(%rsp), %rdx
	callq	memcpy
	movl	$500, %r8d
	movq	%r12, %rcx
	movq	%r14, %rdx
	callq	memcpy
	movq	%rbx, %rdi
	testq	%r15, %r15
	je	.LBB19_30
.LBB19_21:
	cmpq	$9, %rdi
	je	.LBB19_30
	movdqu	%xmm7, 554(%rsp)
	movdqa	%xmm7, 544(%rsp)
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
	movl	$500, %r8d
	movq	%r14, %rcx
	xorl	%edx, %edx
	callq	memset
	cmpq	$251, %r15
	jae	.LBB19_130
	leaq	1(%rdi), %rbx
	movq	192(%rsp), %rax
	movzbl	(%rax,%rdi), %r9d
	leaq	(%rsp,%r15,2), %r10
	addq	$1090, %r10
	xorl	%r15d, %r15d
	movq	%r12, %r11
	.p2align	4
.LBB19_24:
	cmpq	%r10, %r11
	je	.LBB19_29
	movzwl	(%r11), %ecx
	cmpq	$250, %rcx
	jae	.LBB19_45
	addq	$2, %r11
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
	addb	%al, %dl
	movl	%r9d, %r8d
	subb	%dl, %r8b
	jb	.LBB19_24
	movq	%r11, 112(%rsp)
	movq	%r10, 120(%rsp)
	movb	%r9b, 39(%rsp)
	movq	%rbx, 128(%rsp)
	movzbl	1590(%rsp,%rcx), %r11d
	movl	%ecx, %r10d
	andl	$1, %r10d
	movzwl	%cx, %ecx
	shrl	%ecx
	imull	$205, %ecx, %r9d
	movzwl	%r9w, %r9d
	shrl	$10, %r9d
	leal	(%r9,%r9,4), %r9d
	subb	%r9b, %cl
	movzbl	%cl, %r9d
	movzbl	%dl, %ecx
	movq	%rcx, 224(%rsp)
	movzbl	%r8b, %edx
	movq	%rsi, %r8
	movq	%r9, 272(%rsp)
	subq	%r9, %r8
	movq	%r10, 232(%rsp)
	xorq	$1, %r10
	movq	%rdx, 216(%rsp)
	movl	%edx, %ecx
	shrl	%ecx
	cmpq	%r10, %rcx
	cmovbq	%rcx, %r10
	cmpq	%r8, %rsi
	movq	%r8, 264(%rsp)
	cmovbq	%rsi, %r8
	movq	%r8, 208(%rsp)
	movzbl	144(%rsp,%rdi), %r9d
	movzbl	%al, %eax
	leal	(%rax,%rax,4), %eax
	movzbl	%al, %eax
	movq	%rax, 256(%rsp)
	xorl	%eax, %eax
	movq	%rdi, 136(%rsp)
	movq	%r10, 200(%rsp)
	.p2align	4
.LBB19_28:
	movq	%rax, %rcx
	cmpq	%r10, %rax
	adcq	$0, %rax
	movq	%rax, 240(%rsp)
	movq	216(%rsp), %rdx
	subq	%rcx, %rdx
	subq	%rcx, %rdx
	movq	208(%rsp), %rbx
	cmpq	%rbx, %rdx
	movq	%rdx, 288(%rsp)
	cmovbq	%rdx, %rbx
	cmpq	$7, %rdi
	movl	$0, %eax
	cmovaeq	%rax, %rbx
	movq	224(%rsp), %rax
	leaq	(%rax,%rcx,2), %rax
	movq	%rax, 280(%rsp)
	movq	232(%rsp), %rax
	movq	%rcx, 248(%rsp)
	leaq	(%rcx,%rax), %r12
	xorl	%edi, %edi
	.p2align	4
.LBB19_54:
	movq	%rdi, %r10
	cmpq	%rbx, %rdi
	adcq	$0, %rdi
	movq	288(%rsp), %rax
	subq	%r10, %rax
	movq	264(%rsp), %rcx
	subq	%r10, %rcx
	movabsq	$-6148914691236517205, %rdx
	mulq	%rdx
	shrq	%rdx
	cmpq	%rcx, %rdx
	cmovaeq	%rcx, %rdx
	movq	280(%rsp), %rax
	addq	%r10, %rax
	movq	256(%rsp), %rcx
	addq	%r10, %rcx
	leaq	(%rcx,%rcx,4), %r14
	movq	272(%rsp), %rcx
	addq	%r10, %rcx
	addq	%rcx, %r14
	xorl	%ebp, %ebp
	xorl	%r8d, %r8d
	.p2align	4
.LBB19_55:
	cmpq	%rdx, %rbp
	adcq	$0, %r8
	leal	(%rbp,%rbp,2), %ecx
	addl	%eax, %ecx
	subb	%r9b, %cl
	movzbl	%cl, %ecx
	movl	$0, %esi
	cmovbl	%esi, %ecx
	addb	%r11b, %cl
	movzbl	%cl, %r13d
	movl	$255, %ecx
	cmovbl	%ecx, %r13d
	leaq	(%r14,%rbp), %rcx
	leaq	(%r12,%rcx,2), %rcx
	cmpq	$249, %rcx
	ja	.LBB19_131
	movzbl	320(%rsp,%rcx), %esi
	cmpb	%sil, %r13b
	jae	.LBB19_57
	cmpb	$-1, %sil
	jne	.LBB19_62
	cmpq	$249, %r15
	ja	.LBB19_132
	movw	%cx, 584(%rsp,%r15,2)
	incq	%r15
.LBB19_62:
	movb	%r13b, 320(%rsp,%rcx)
.LBB19_57:
	cmpq	%rdx, %rbp
	jae	.LBB19_52
	movq	%r8, %rbp
	cmpq	%rdx, %r8
	jbe	.LBB19_55
.LBB19_52:
	cmpq	%rbx, %r10
	jae	.LBB19_47
	cmpq	%rbx, %rdi
	jbe	.LBB19_54
.LBB19_47:
	movq	200(%rsp), %r10
	cmpq	%r10, 248(%rsp)
	jae	.LBB19_48
	movq	240(%rsp), %rax
	cmpq	%r10, %rax
	movq	136(%rsp), %rdi
	jbe	.LBB19_28
	leaq	1090(%rsp), %r12
	leaq	584(%rsp), %r14
	movq	104(%rsp), %rsi
	jmp	.LBB19_51
.LBB19_48:
	leaq	1090(%rsp), %r12
	leaq	584(%rsp), %r14
	movq	104(%rsp), %rsi
	movq	136(%rsp), %rdi
.LBB19_51:
	movq	128(%rsp), %rbx
	movzbl	39(%rsp), %r9d
	movq	120(%rsp), %r10
	movq	112(%rsp), %r11
	jmp	.LBB19_24
	.p2align	4
.LBB19_30:
	movw	$-1, 56(%rsp)
	movq	$-1, 48(%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	80(%rsp), %r15
	movq	40(%rsp), %r8
	.p2align	4
.LBB19_31:
	cmpq	%rsi, %rcx
	adcq	$0, %rdx
	leaq	(%rcx,%rcx), %rax
	cmpq	$249, %rax
	ja	.LBB19_63
	cmpq	$4, %rcx
	ja	.LBB19_65
	movzwl	1590(%rsp,%rcx,2), %eax
	movw	%ax, 48(%rsp,%rcx,2)
	cmpq	%rsi, %rcx
	jae	.LBB19_35
	movq	%rdx, %rcx
	cmpq	%rsi, %rdx
	jbe	.LBB19_31
.LBB19_35:
	movq	176(%rsp), %rdx
	movzbl	8(%rdx), %eax
	leaq	1092(%rsp), %rcx
	movb	%al, 7(%rcx)
	movq	(%rdx), %rax
	movq	%rax, -1(%rcx)
	movzbl	38(%rsp), %eax
	movb	%al, 1090(%rsp)
	movq	32(%r8), %rcx
	movq	40(%r8), %rdx
	movq	%r12, %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRThAhj9_EECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	40(%rsp), %rcx
	cmpq	$0, 16(%rcx)
	je	.LBB19_36
.LBB19_37:
	movq	(%rcx), %r13
	movq	8(%rcx), %r8
	movq	%rax, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	movzbl	1099(%rsp), %r10d
	movq	1091(%rsp), %r11
	xorl	%ebx, %ebx
	xorl	%edi, %edi
.LBB19_38:
	andq	%r8, %rax
	movdqu	(%r13,%rax), %xmm1
	movdqa	%xmm1, %xmm2
	pcmpeqb	%xmm0, %xmm2
	pmovmskb	%xmm2, %r14d
	testl	%r14d, %r14d
	je	.LBB19_42
.LBB19_39:
	rep		bsfl	%r14d, %esi
	addq	%rax, %rsi
	andq	%r8, %rsi
	negq	%rsi
	leaq	(%rsi,%rsi,4), %r15
	movzbl	38(%rsp), %ecx
	cmpb	-20(%r13,%r15,4), %cl
	jne	.LBB19_41
	movzbl	-11(%r13,%r15,4), %esi
	movq	-19(%r13,%r15,4), %r12
	xorq	%r11, %r12
	xorl	%r10d, %esi
	movzbl	%sil, %esi
	orq	%r12, %rsi
	leaq	1090(%rsp), %r12
	je	.LBB19_73
.LBB19_41:
	leal	-1(%r14), %esi
	andw	%r14w, %si
	movl	%esi, %r14d
	movq	80(%rsp), %r15
	jne	.LBB19_39
	.p2align	4
.LBB19_42:
	cmpq	$1, %rbx
	je	.LBB19_67
	pmovmskb	%xmm1, %r9d
	testl	%r9d, %r9d
	je	.LBB19_44
	rep		bsfl	%r9d, %r9d
	addq	%rax, %r9
	andq	%r8, %r9
.LBB19_67:
	pcmpeqb	%xmm7, %xmm1
	pmovmskb	%xmm1, %esi
	testl	%esi, %esi
	jne	.LBB19_70
	movl	$1, %ebx
	jmp	.LBB19_69
.LBB19_44:
	xorl	%ebx, %ebx
.LBB19_69:
	addq	%rdi, %rax
	addq	$16, %rax
	addq	$16, %rdi
	jmp	.LBB19_38
	.p2align	4
.LBB19_19:
	leaq	(%rcx,%rdi,4), %rax
	movzwl	-2(%rax), %ecx
	movw	%cx, 56(%rsp)
	movq	-10(%rax), %rax
	movq	%rax, 48(%rsp)
	movq	40(%rsp), %rcx
	jmp	.LBB19_75
	.p2align	4
.LBB19_73:
	leaq	(,%r15,4), %rax
	addq	%r13, %rax
	movzwl	56(%rsp), %ecx
	movw	%cx, -2(%rax)
	movq	48(%rsp), %rcx
	movq	%rcx, -10(%rax)
	movq	80(%rsp), %r15
	movq	40(%rsp), %rcx
	jmp	.LBB19_74
.LBB19_70:
	movzbl	(%r13,%r9), %eax
	testb	%al, %al
	jns	.LBB19_71
.LBB19_72:
	movzwl	1098(%rsp), %r10d
	movw	%r10w, 592(%rsp)
	movq	1090(%rsp), %r10
	movq	%r10, 584(%rsp)
	movzwl	56(%rsp), %r10d
	leaq	594(%rsp), %rcx
	movw	%r10w, 8(%rcx)
	movq	48(%rsp), %r10
	movq	%r10, (%rcx)
	andb	$1, %al
	leaq	-16(%r9), %r10
	andq	%r8, %r10
	movb	%dl, (%r13,%r9)
	movb	%dl, 16(%r13,%r10)
	movq	40(%rsp), %rcx
	movdqu	16(%rcx), %xmm0
	movzbl	%al, %eax
	movd	%eax, %xmm1
	shufps	$228, %xmm6, %xmm1
	psubq	%xmm1, %xmm0
	movdqu	%xmm0, 16(%rcx)
	negq	%r9
	leaq	(%r9,%r9,4), %rax
	movl	600(%rsp), %edx
	movl	%edx, -4(%r13,%rax,4)
	movdqu	584(%rsp), %xmm0
	movdqu	%xmm0, -20(%r13,%rax,4)
.LBB19_74:
	leaq	584(%rsp), %r14
.LBB19_75:
	movw	$-1, 592(%rsp)
	movq	$-1, 584(%rsp)
	movq	112(%rcx), %rcx
	xorl	%eax, %eax
	.p2align	4
.LBB19_76:
	xorl	%edx, %edx
	movq	%rcx, %r11
	subq	%rax, %r11
	seta	%r8b
	cmpq	$4, %rax
	ja	.LBB19_79
	movzbl	64(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_80
	xorl	%r10d, %r10d
	xorl	%edi, %edi
	.p2align	4
.LBB19_94:
	cmpq	%r11, %r10
	adcq	$0, %rdi
	cmpq	$4, %r10
	ja	.LBB19_133
	leaq	(%r10,%rax), %r9
	movzbl	48(%rsp,%r10,2), %ebp
	cmpq	$4, %r9
	ja	.LBB19_100
	cmpb	$-1, %bpl
	je	.LBB19_98
	addb	%bl, %bpl
	movzbl	%bpl, %esi
	movl	$255, %ebp
	cmovbl	%ebp, %esi
	movzbl	584(%rsp,%r9,2), %ebp
	cmpb	%bpl, %sil
	cmovael	%ebp, %esi
	movb	%sil, 584(%rsp,%r9,2)
.LBB19_98:
	movzbl	49(%rsp,%r10,2), %ebp
	cmpb	$-1, %bpl
	je	.LBB19_92
	addb	%bl, %bpl
	movzbl	%bpl, %esi
	movl	$255, %ebp
	cmovbl	%ebp, %esi
	movzbl	585(%rsp,%r9,2), %ebp
	cmpb	%bpl, %sil
	cmovael	%ebp, %esi
	movb	%sil, 585(%rsp,%r9,2)
	jmp	.LBB19_92
	.p2align	4
.LBB19_100:
	cmpb	$-1, %bpl
	jne	.LBB19_102
	cmpb	$-1, 49(%rsp,%r10,2)
	jne	.LBB19_102
.LBB19_92:
	cmpq	%r11, %r10
	jae	.LBB19_80
	movq	%rdi, %r10
	cmpq	%r11, %rdi
	jbe	.LBB19_94
.LBB19_80:
	movzbl	65(%rsp,%rax,2), %ebx
	cmpb	$-1, %bl
	je	.LBB19_89
	xorl	%r10d, %r10d
	xorl	%edi, %edi
	.p2align	4
.LBB19_82:
	cmpq	%r11, %r10
	adcq	$0, %rdi
	cmpq	$4, %r10
	ja	.LBB19_133
	leaq	(%r10,%rax), %r9
	movzbl	48(%rsp,%r10,2), %ebp
	cmpq	$5, %r9
	jae	.LBB19_84
	cmpb	$-1, %bpl
	je	.LBB19_87
	addb	%bl, %bpl
	movzbl	%bpl, %esi
	movl	$255, %ebp
	cmovbl	%ebp, %esi
	movzbl	585(%rsp,%r9,2), %ebp
	cmpb	%bpl, %sil
	cmovael	%ebp, %esi
	movb	%sil, 585(%rsp,%r9,2)
	jmp	.LBB19_87
	.p2align	4
.LBB19_84:
	cmpb	$-1, %bpl
	jne	.LBB19_102
.LBB19_87:
	cmpq	%r11, %r10
	jae	.LBB19_89
	movq	%rdi, %r10
	cmpq	%r11, %rdi
	jbe	.LBB19_82
.LBB19_89:
	cmpq	%rax, %rcx
	jbe	.LBB19_91
	movb	%r8b, %dl
	addq	%rdx, %rax
	cmpq	%rcx, %rax
	jbe	.LBB19_76
	jmp	.LBB19_91
.LBB19_36:
	movl	$1, %edx
	movq	160(%rsp), %r8
	movb	$1, %r9b
	movq	%rax, %rsi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTThAhj9_EAAhj2_j5_EE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_BY_NtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	40(%rsp), %rcx
	movq	%rsi, %rax
	jmp	.LBB19_37
.LBB19_71:
	movdqa	(%r13), %xmm0
	pmovmskb	%xmm0, %eax
	rep		bsfl	%eax, %r9d
	movzbl	(%r13,%r9), %eax
	jmp	.LBB19_72
.LBB19_131:
	leaq	anon.da267e598a0a058d3d272408d5046ada.138(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_108:
	addq	%r10, %rcx
	movzbl	-1(%rcx), %eax
.LBB19_109:
	movaps	1840(%rsp), %xmm6
	movaps	1856(%rsp), %xmm7
	.seh_startepilogue
	addq	$1880, %rsp
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
.LBB19_132:
	leaq	anon.da267e598a0a058d3d272408d5046ada.139(%rip), %r8
	movl	$250, %edx
	movq	%r15, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_7:
	cmpq	$5, %rcx
	jae	.LBB19_8
	movzbl	65(%rsp,%rcx,2), %esi
	cmpq	$0, 104(%r11)
	jne	.LBB19_113
	movzbl	147(%r11), %r8d
	movups	anon.da267e598a0a058d3d272408d5046ada.23+11(%rip), %xmm0
	movups	%xmm0, 611(%rsp)
	movdqu	anon.da267e598a0a058d3d272408d5046ada.23(%rip), %xmm0
	movdqu	%xmm0, 600(%rsp)
	xorl	%eax, %eax
	xorl	%edx, %edx
	xorl	%r9d, %r9d
	.p2align	4
.LBB19_105:
	movzbl	600(%rsp,%r9), %r10d
	movl	%r10d, %ecx
	subb	$9, %cl
	cmpb	$9, %cl
	movl	$2, %ecx
	sbbl	$0, %ecx
	cmpb	$9, %r10b
	movl	$0, %r11d
	cmovaeq	%rcx, %r11
	cmpl	%r8d, %r11d
	jne	.LBB19_106
	incq	%r9
	cmpq	$27, %r9
	jne	.LBB19_105
	jmp	.LBB19_111
.LBB19_106:
	movzbl	%r10b, %ecx
	cmpb	$26, %r10b
	movq	40(%rsp), %r11
	ja	.LBB19_134
	movzbl	(%r15,%rcx), %ecx
	movl	%ecx, %r10d
	shrb	%r10b
	addb	%r10b, %al
	andb	$1, %cl
	addb	%cl, %dl
	cmpq	$26, %r9
	leaq	1(%r9), %r9
	jne	.LBB19_105
	jmp	.LBB19_112
.LBB19_111:
	movq	40(%rsp), %r11
.LBB19_112:
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
.LBB19_113:
	movq	80(%r11), %rcx
	movq	88(%r11), %rdx
	movq	%r15, %r8
	callq	_RINvYNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateNtNtCs9OCfJ4KTcUD_4core4hash11BuildHasher8hash_oneRAhj1b_ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	40(%rsp), %rdx
	movq	%rax, %r10
	cmpq	$0, 64(%rdx)
	je	.LBB19_114
.LBB19_115:
	movq	48(%rdx), %rcx
	movq	56(%rdx), %r8
	movq	%r10, %rdx
	shrq	$57, %rdx
	movd	%edx, %xmm0
	punpcklbw	%xmm0, %xmm0
	pshuflw	$0, %xmm0, %xmm0
	pshufd	$68, %xmm0, %xmm0
	xorl	%edi, %edi
	pcmpeqd	%xmm1, %xmm1
	xorl	%r11d, %r11d
.LBB19_116:
	andq	%r8, %r10
	movdqu	(%rcx,%r10), %xmm2
	movdqa	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb	%xmm3, %ebx
	testl	%ebx, %ebx
	movl	%esi, %eax
	je	.LBB19_119
.LBB19_117:
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
	je	.LBB19_129
	leal	-1(%rbx), %ebp
	andw	%bx, %bp
	movl	%ebp, %ebx
	jne	.LBB19_117
.LBB19_119:
	cmpq	$1, %rdi
	je	.LBB19_123
	pmovmskb	%xmm2, %r9d
	testl	%r9d, %r9d
	je	.LBB19_121
	rep		bsfl	%r9d, %r9d
	addq	%r10, %r9
	andq	%r8, %r9
.LBB19_123:
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %edi
	testl	%edi, %edi
	jne	.LBB19_126
	movl	$1, %edi
	jmp	.LBB19_125
.LBB19_121:
	xorl	%edi, %edi
.LBB19_125:
	addq	%r11, %r10
	addq	$16, %r10
	addq	$16, %r11
	movq	80(%rsp), %r15
	jmp	.LBB19_116
.LBB19_129:
	addq	%r14, %rcx
	movb	%al, -1(%rcx)
	jmp	.LBB19_109
.LBB19_126:
	movzbl	(%rcx,%r9), %r10d
	testb	%r10b, %r10b
	jns	.LBB19_127
.LBB19_128:
	andb	$1, %r10b
	leaq	-16(%r9), %r11
	andq	%r8, %r11
	movb	%dl, (%rcx,%r9)
	movb	%dl, 16(%rcx,%r11)
	movq	40(%rsp), %r8
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
	movdqu	(%r15), %xmm0
	movdqu	11(%r15), %xmm1
	movdqu	%xmm1, -17(%rcx,%rdx)
	movdqu	%xmm0, -28(%rcx,%rdx)
	movb	%al, -1(%rcx,%rdx)
	jmp	.LBB19_109
.LBB19_130:
	leaq	anon.da267e598a0a058d3d272408d5046ada.142(%rip), %r9
	movl	$250, %r8d
	xorl	%ecx, %ecx
	movq	%r15, %rdx
	callq	_RNvNtNtCs9OCfJ4KTcUD_4core5slice5index16slice_index_fail
	ud2
.LBB19_133:
	leaq	anon.da267e598a0a058d3d272408d5046ada.21(%rip), %r8
	movl	$5, %edx
	movq	%r10, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_102:
	leaq	anon.da267e598a0a058d3d272408d5046ada.22(%rip), %r8
	movl	$5, %edx
	movq	%r9, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_45:
	leaq	anon.da267e598a0a058d3d272408d5046ada.137(%rip), %r8
	movl	$250, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_79:
	leaq	anon.da267e598a0a058d3d272408d5046ada.20(%rip), %r8
	movl	$5, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_114:
	leaq	48(%rdx), %rcx
	leaq	80(%rdx), %r8
	movl	$1, %edx
	movb	$1, %r9b
	movq	%r10, %rdi
	callq	_RINvMs6_NtCs5zImONXtN8M_9hashbrown3rawINtB6_8RawTableTAhj1b_hEE14reserve_rehashNCINvNtB8_3map11make_hasherBQ_hNtNtNtCsbSQC797JefK_3std4hash6random11RandomStateE0ECs4DgSkW5ScxR_19sichuan_mahjong_sim
	movq	40(%rsp), %rdx
	movq	%rdi, %r10
	jmp	.LBB19_115
.LBB19_65:
	leaq	anon.da267e598a0a058d3d272408d5046ada.141(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_63:
	leaq	anon.da267e598a0a058d3d272408d5046ada.140(%rip), %r8
	movl	$250, %edx
	movq	%rax, %rcx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_127:
	movdqa	(%rcx), %xmm0
	pmovmskb	%xmm0, %r9d
	rep		bsfl	%r9d, %r9d
	movzbl	(%rcx,%r9), %r10d
	jmp	.LBB19_128
.LBB19_8:
	leaq	anon.da267e598a0a058d3d272408d5046ada.19(%rip), %r8
	movl	$5, %edx
	callq	_RNvNtCs9OCfJ4KTcUD_4core9panicking18panic_bounds_check
	ud2
.LBB19_134:
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
