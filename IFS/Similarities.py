import numpy as np

'''
	Similarity measures for (intuitionistic) fuzzy sets
	========================================================================
	© Visual Computing Group, HUMAIN-Lab, Eastern Macedonia and Thrace Institute of Technology
	01-Mar-2017 - Release
'''

def __Simularities__():
    # print('\n'
    #     'dengChunSimilarity \n'
    #     'liangShiSimilarity\n'
    #     'parkKwunLimSimilarity\n'
    #     'mitchellSimilarity\n'
    #     'A_vulHungLinSimilarity\n'
    #     'hungYangSimilarity\n'
    #     'yenSimilarity\n'
    #     'hwangYangSimilarity\n'
    #     'hungYangSimilarity2\n'
    #     'zhangFuSimilarity\n'
    #     'hungYangSimilarity3\n'
    #     'chenSimilarity\n'
    #     'hungYangDistance4\n'
    #     'hongKimSimilarity\n'
    #     'chenSimilarity2\n'
    #     'lA_mSimilarity\n'
    #     'iancuSimilarity\n'
    #     'songWangLeiXueSimilarity\n'
    #     'intarapaiboonSimilarity\n'
    #     'dengA_viangFuSimilarity\n'
    #     'nguyenSimilarity\n'
    #     'chenChengLanSimilarity\n'
    #     'muthuKrishSimilarity\n')
    return ['dengChunSimilarity',
			'liangShiSimilarity',
			'parkKwunLimSimilarity',
			'mitchellSimilarity',
			'A_vulHungLinSimilarity',
			'hungYangSimilarity',
			'yenSimilarity',
			'hwangYangSimilarity',
			'hungYangSimilarity2',
			'zhangFuSimilarity',
			'hungYangSimilarity3',
			'chenSimilarity',
			'hungYangDistance4',
			'hongKimSimilarity',
			'chenSimilarity2',
			'lA_mSimilarity',
			'iancuSimilarity',
			'songWangLeiXueSimilarity',
			'intarapaiboonSimilarity',
			'dengA_viangFuSimilarity',
			'nguyenSimilarity',
			'chenChengLanSimilarity',
			'muthuKrishSimilarity']

#####################################################################################################################

def similarity(similarityMeasure, A, B, p = 1, w = None, type = None, omegas = [0.5 , 0.3 , 0.2], a = 1 , b = 0, c = 0, lamda = 1):
    if similarityMeasure == 'dengChunSimilarity':
        return IFS_Sim01(A , B , p , w)
    elif similarityMeasure == 'liangShiSimilarity':
        return IFS_Sim02(A, B, p=p,type=type, w=w, omegas=omegas)
    elif similarityMeasure == 'parkKwunLimSimilarity':
        return IFS_Sim03(A, B ,p, w)
    elif similarityMeasure == 'mitchellSimilarity':
        return IFS_Sim04(A,B ,p, w)
    elif similarityMeasure == 'julHungLinSimilarity':
        return IFS_Sim05(A, B ,p, w, type)
    elif similarityMeasure == 'hungYangSimilarity':
        return IFS_Sim06(A, B,type, w)
    elif similarityMeasure == 'yenSimilarity':
        return IFS_Sim07(A,B, w)
    elif similarityMeasure == 'hwangYangSimilarity':
        return IFS_Sim08(A,B,w)
    elif similarityMeasure == 'hungYangSimilarity2':
        return IFS_Sim09(A,B,a,type)
    elif similarityMeasure == 'zhangFuSimilarity':
        return IFS_Sim10(A ,  B)
    elif similarityMeasure == 'hungYangSimilarity3':
        return IFS_Sim11(A, B ,type)
    elif similarityMeasure == 'chenSimilarity':
        return IFS_Sim12(A, B,w)
    elif similarityMeasure == 'hungYangDistance4':
        return IFS_Sim13(A,B,p,type)
    elif similarityMeasure == 'hongKimSimilarity':
        return IFS_Sim14(A, B , w, a,b,c)
    elif similarityMeasure == 'chenSimilarity2':
        return IFS_Sim15(A, B,w, a = 1 , b = 0 , c = 0)
    elif similarityMeasure == 'liuSimilarity':
        return IFS_Sim16(A, B,p, w, a,b,c)
    elif similarityMeasure == 'iancuSimilarity':
        return IFS_Sim17(A,B, type, lamda)
    elif similarityMeasure == 'songWangLeiXueSimilarity':
        return IFS_Sim18(A,B, w)
    elif similarityMeasure == 'intarapaiboonSimilarity':
        return IFS_Sim19(A , B)
    elif similarityMeasure == 'dengA_viangFuSimilarity':
        return IFS_Sim20(A , B)
    elif similarityMeasure == 'nguyenSimilarity':
        return IFS_Sim21(A,B)
    elif similarityMeasure == 'chenChengLanSimilarity':
        return IFS_Sim22(A,B)
    elif similarityMeasure == 'muthuKrishSimilarity':
        return IFS_Sim23(A , B)
    else:
        print('Unknown similarity measure input.')
        return None

#####################################################################################################################

def IFS_Sim01(A, B,p, w):
    '''
	IFS_SIM01: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by L. Dengfeng, C. Chuntian, from the related article 
	"New similarity measures of intuitionistic fuzzy sets and application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: power parameter. Must be equal or higher than 1.
		w: weights for the computed similarity. If 0 then the non-weighted
		similarity is computed.
	OUTPUT:
		S: Similarity measure.
    '''
    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:,0])
    B_v = np.array(B[:,1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not(A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    w = np.array(w)
	
    if w is None:
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
	
    if w.__len__() == 1 and w != 0:
        if (w.size != A.size or w.ndim != 1):
            print('Weight parameter dimensions must be 1-D Nx1 matrix.')
            return
        if np.sum(w < 0 )!= 0 or np.sum( w > 1) != 0 :
            print('Weight values must be 0 <= w <= 1.')
            return
        if w.sum() != 1:
            print('Sum of weights must be equal to 1')
            return
    if p < 1:
        print('p parameter must be equal or higher than 1.')
        return


    fA = (A_m + 1 - A_v)/2.0
    fB = (B_m + 1 - B_v)/2.0
    n = A.__len__()
    if (w.__len__() == 1 and w != 0):
        return 1 - 1/(n**(1/p)) * (np.sum(np.absolute(fA - fB)**p)**(1/p))
    else:
        return 1 - np.power(np.sum(w * np.sum(np.power(np.absolute(fA - fB),p))),(1.0/float(p)))

##############################################################################################################################################

def IFS_Sim02(A, B, p, type, w = np.array([]), omegas = [0.5 , 0.3 , 0.2] ):#, varagrin):
    '''
	IFS_SIM02: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Z. Liang, P. Shi, from the related article 
	"Similarity measures on intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed similarity: 'e', 's' or 'h'.
		p: power parameter. Must be equal or higher than 1.
		w: weights for the computed similarity. Used only for type = 'h'. Can
		be ommited.
		omegas: matrix containing values that belong to [0,1]. The sum of the 
		matrix must be equal to 1. Used only for type = 'h'.
	OUTPUT:
		S: Similarity measure.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    n = len(A)


    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if p < 1 and not isinstance(p, int):
        print('p parameter must be an integer > 0')
        return



    if type == 'h':
        if w is None : 
            w = np.full(A.shape[0] , 1 / float(A.shape[0]))
        elif w.size == 0 or w.__len__() == 1 :
            w = np.ones(n)
        elif w.size != A.size:
            w = np.array(w)

            if np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
                print('Weight values must be 0 <= w <= 1.')
                return

            if (w.size != A.size ):
                print('Weight parameter dimensions must be 1-D Nx1 matrix.')
                return

            if w.sum() != 1:
                print('Sum of weights must be equal to 1')
                return
        omegas = np.array(omegas)

        if not omegas.shape == (3,) :

            print('Omegas parameter must be a 3x1 matrix.')
            return

    if (type == 'e'):
        mDif = np.absolute(A_m - B_m)
        vDif = np.absolute(A_v - B_v)

        D = np.power((np.sum(np.power((mDif/2.0 + vDif/2.0),p))),(1.0/float(p)))
        return 1 - 1.0/float(np.power(n,(1.0/p))) * D
    elif (type == 's'):
        mA = (A_m + 1 - A_v) / 2.0
        mB = (B_m + 1 - B_v) / 2.0
        mA1 = (A_m + mA) / 2.0
        mB1 = (B_m + mB) / 2.0
        mA2 = (mA + 1 - A_v) / 2.0
        mB2 = (mB + 1 - B_v) / 2.0
        fS1 = np.absolute(mA1 - mB1) / 2.0
        fS2 = np.absolute(mA2 - mB2) / 2.0
        D = np.power((np.sum(np.power((fS1 + fS2), p))) , (1.0 / p))
        return 1 - 1.0 / float(np.power(n, (1.0 / p))) * D
    elif (type == 'h'):
        f1 = np.absolute(A_m-B_m) / 2.0 + np.absolute(((1-A_v)/2.0)-(1.0/2.0))
        mA = (A_m + 1 - A_v) / 2.0
        mB = (B_m + 1 - B_v) / 2.0
        f2 = np.absolute(mA - mB) / 2.0
        iA = mA - A_m
        iB = mB - B_m
        f3 = np.maximum(iA, iB - np.minimum(iA, iB))
        D = np.power((np.sum(w * (np.power((omegas[0] * f1 + omegas[1] * f2 + omegas[2] * f3),p)))),(1.0/p))
        return 1 - D

    else:
        print('Distance type parameter must be e, s or h')
        return

########################################################################################################################

def IFS_Sim03(A, B ,p, w):
    '''
	IFS_SIM03: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by A_v.H. Park, A_v.S. Park, Y.C. Kwun, K.M. Lim, from 
	the related article "New similarity measures on intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic  fuzzy set of the ideally segmented image.
    '''
    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])


    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if len(A[0]) != 3  or not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx3 matrices.')
        return
    w = np.array(w)

    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return

    if p < 1:
        print('p parameter must be equal or higher than 1.')
        return

    D = np.sum(w * np.power((np.absolute(A_m - B_m)/2.0 + np.absolute(A_v-B_v)/2.0 + np.absolute(A_p - B_p)/2.0),p))

    return 1 - (np.power(D , (1.0/p)))

########################################################################################################################

def IFS_Sim04(A,B ,p, w):
    '''
	IFS_SIM04: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by H.B. Mitchell, from the related article 
	"On the Dengfeng-Chuntian similarity measure and its application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])


    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:,1])


    w = np.array(w)

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    w = np.array(w)
	
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.__len__() != A.shape[0]*A.shape[1] or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return

    if p < 1 and not isinstance(p , int):
        print('p parameter must be integer')
        return

    D_m = 1 - np.power(np.sum(w * np.power(np.absolute(A_m - B_m),p)),(1.0/p))
    D_f = 1 - np.power(np.sum(w * np.power(np.absolute(A_v - B_v),p)),(1.0/p))

    return (D_m + D_f)/2.0

########################################################################################################################

def IFS_Sim05(A, B ,p, w, type):
    '''
	IFS_SIM05: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by P. A_vulian, K.C. Hung, S.A_v. Lin, from the related article 
	"On the Mitchell similarity measure and its application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		w: weights for the computed similarity.
		type: Type of computed similarity: 1 or 2.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''


    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    w = np.array(w)

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
    w = np.array(w)
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return
    
    if p < 1 and not isinstance(p, int):
        print('p parameter must be integer')
        return

    D_m = np.sum(w *  np.power(np.absolute(A_m - B_m), p))
    D_f = np.sum(w * np.power(np.absolute(A_v - B_v), p))



    if (type == 1):
        #return 1 - ((D_m * 0.6) ** (1 / float(p))) - ((D_f * 0.6) ** (1 / float(p)))
        return 1 - np.sum(np.power((D_m * 0,6),(1.0/p)) - np.power((D_f * 0.6),(1.0/p)))
    else:
        return 1 - np.sum(np.power(D_m,(1.0/p)) - np.power(D_f,(1.0/p)))

########################################################################################################################

def IFS_Sim06(A, B,type, w):
    '''
	IFS_SIM06: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung, M.S. Yang, from the related article 
	"Similarity measures of intuitionistic fuzzy sets based on Hausdorff similarity"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed similarity: 'l', 'e' or 'c'
		w: weights for the computed similarity. If 0 then the non-weighted
		similarity is computed.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''


    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return
	
	
	
    if (np.sum(w) == 0):
        D = np.sum(np.maximum(np.absolute(A_m - B_m),np.absolute((1 - A_v) - (1 - B_v)))/float(len(A)))
    else:
        D = np.sum(w * (np.maximum(np.absolute(A_m - B_m), np.absolute((1 - A_v) - (1 - B_v))) / float(len(A))))

    if type == 'l':
        return 1 - D
    elif type == 'e':
        tp = np.exp(-1)
        return (np.exp(-D) - tp) / float(1 - tp)
    else:
        return (1 - D) / float(1 + D)

########################################################################################################################

def IFS_Sim07(A,B, w):
    '''
	IFS_SIM07: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by A_v. Yen, from the related article 
	"Cosine similarity measures for intuitionistic fuzzy sets and their applications"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])


    w = np.array(w)

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
    w = np.array(w)
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return


    num = A_m * B_m + A_v * B_v
    denum = np.sqrt(np.power(A_m,2.0) + np.power(A_v,2.0)) * np.sqrt(np.power(B_m,2.0) + np.power(B_v,2.0))

    return np.sum(w * (num / np.array(denum).astype("float")))#ret_sum

########################################################################################################################

def IFS_Sim08(A, B,w):
    '''
	IFS_SIM08: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by C.M.  Hwang and M.S. Yang, from the related article 
	"Modified cosine similarity measure between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)

    A_m = np.array(A[:, 0]) # kai num1
    A_v = np.array(A[:, 1])


    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    w = np.array(w)

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
    w = np.array(w)

    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return
    elif w.sum() != 1:
        print('Sum of weights must be equal to 1')
        return

    num = A_m * B_m + A_v * B_v
    denum = np.sqrt(np.power(A_m,2.0) + np.power(A_v,2.0)) * np.sqrt(np.power(B_m,2.0) + np.power(B_v,2.0))
    C1 = np.zeros(len(A))

    C1_mask = (denum != 0 )

    C1[C1_mask] = num[C1_mask] / (np.array(denum[C1_mask]).astype("float"))

    fA = (1 + A_m - A_v) / 2.0
    fB = (1 + B_m - B_v) / 2.0

    num = fA * fB + A_m * B_m
    denum = np.sqrt(np.square(fA)+np.square(A_v)) * np.sqrt(np.square(fB) + np.square(A_v))

    C2 = num / np.array(denum).astype("float")

    num = (1 - A_m) * (1 - B_m) * (1 - A_v) * (1 - B_v)
    denum = np.sqrt(np.square((1 - A_m)) + np.square((1 - A_v))) * np.sqrt(np.square((1 - B_m)) + np.square((1 - B_v)))

    C3 = num / np.array(denum).astype("float")

    return  np.sum(w * (C1 + C2 + C3) / 3.0)

########################################################################################################################

def IFS_Sim09(A, B ,type ,a = 1):
    '''
	IFS_SIM09: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung, M.S. Yang, from the related article 
	"On the A_v-divergence of intuitionistic fuzzy sets with its applications to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree
		a: case of divergence measure. Positive integer.
		type: Type of computed similarity: 'l', 'e' or 'c'.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A = np.array(A)

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    n = len(A)

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if a < 1 and not isinstance(a, int):
        print('a parameter must be integer')
        return

    if a == 1:
        U_a = np.log(2.0)

        LABm1 = (A_m + B_m)

        msk_n = ~((A_m + B_m) == 0)

        LABm1[msk_n] *=  np.log((A_m[msk_n] + B_m[msk_n]) / 2.0)

        LABm2 = A_m

        msk_n = ~(A_m == 0)
        LABm2[msk_n] *= np.log(A_m[msk_n])


        LABm3 = B_m
        msk_n = ~(B_m == 0)
        LABm3[msk_n]*= np.log(B_m[msk_n])

        LABv1 = (A_v + B_v)

        msk_n = ~((A_v + B_v) == 0)

        LABv1[msk_n] *= np.log((A_v[msk_n] + B_v[msk_n]) / 2.0)

        LABv2 = A_v

        msk_n = ~(A_v == 0)
        LABv2[msk_n] *= np.log(A_v[msk_n])

        LABv3 = B_v
        msk_n = ~(B_v == 0)
        LABv3[msk_n] *= np.log(B_v[msk_n])

        LABp1 = (A_p + B_p)

        msk_n = ~((A_p + B_p) == 0)

        LABp1[msk_n] *= np.log((A_p[msk_n] + B_p[msk_n]) / 2.0)

        LABp2 = A_p

        msk_n = ~(A_p == 0)
        LABp2[msk_n] *= np.log(A_p[msk_n])

        LABp3 = B_p
        msk_n = ~(B_p == 0)
        LABp3[msk_n] *= np.log(B_p[msk_n])

        LABm = LABm1 - LABm2 - LABm3
        LABv = LABv1 - LABv2 - LABv3
        LABp = LABp1 - LABp2 - LABp3


        A_v = -0.5 * np.sum(LABm + LABv + LABp)
    else:
        U_a = 1.0 / (float(a) - 1.0) * (1 - 1 / float(2.0 ** (a - 1.0)))
        TABm = (np.power(((A_m + B_m) / 2.0), a)) - 0.5 * (np.power(A_m , a) + np.power(B_m , a))
        TABv = (np.power(((A_v + B_v) / 2.0), a)) - 0.5 * (np.power(A_v , a) + np.power(B_v , a))
        TABp = (np.power(((A_p + B_p) / 2.0), a)) - 0.5 * (np.power(A_p , a) + np.power(B_p , a))
        TAB = TABm + TABv + TABp

        #A_v = np.sum(1.0/(a-1.0) * TAB)
        A_v = np.sum(np.divide(1.0 , np.multiply((a-1.0),TAB)))

    A_v_a = A_v / float(n)


    if (type == 'l'):
        return (U_a - A_v_a) / float(U_a)
    elif (type == 'e'):
        return (np.exp(-A_v_a) - np.exp(-U_a)) / float(1 - np.exp(-U_a))
    elif (type == 'c'):
        np.seterr(all='ignore')
        return (U_a - A_v_a) / np.array((1 + A_v_a) * U_a).astype("float")

########################################################################################################################


def IFS_Sim10(A ,  B):
    '''
	IFS_SIM10: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by C. Zhang, H. Fu, from the related article 
	"Similarity measures on three kinds of fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    n = len(A)

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if len(A[0]) != 3 or not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx3 matrices.')
        return

    dA = (A_m + A_p) * A_m
    aA = (A_v + A_p) * A_v

    dB = (B_m + B_p) * B_m
    aB = (B_v + B_p) * B_v


    D = np.sum(np.absolute(dA - dB) + np.absolute(aA - aB))


    return 1.0 - D / (2.0 * n)

########################################################################################################################

def IFS_Sim11(A, B ,type):
    '''
	IFS_SIM11: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by W.L. Hung and M.S. Yang, from the related article 
	"On similarity measures between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D array containing in the first dimension the membership values
		and in the second the non-membership values.
		type: Type of computed similarity: 'w1', 'w2', 'pk1', 'pk2', 'pk3', 'new1'
		or 'new2'.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''


    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if type == 'w1':

        num = np.minimum(A_m , B_m) + np.minimum(A_v , B_v)
        denom = np.maximum(A_m , B_m) + np.maximum(A_v , B_v)
        yy = np.divide(num , denom.astype(float))

        D = np.sum(yy)

        return D / float(len(A))
    elif type == 'w2':

        n = len(A)
        ptr2 = np.absolute(A_m - B_m)+ np.absolute(A_v - B_v)
        tmp = ptr2 * 0.5
        D = np.sum(np.subtract( 1,tmp))
        return D / float(n)

    elif type == 'pk1':
        num = np.minimum(A_m,B_m) + np.minimum(A_v,B_v)
        denom = np.maximum(A_m,B_m) + np.maximum(A_v,B_v)
        D1 = np.sum(num)
        D2 = np.sum(denom)

        return D1 / float(D2)
    elif type == 'pk2':

        dif_m = np.absolute(A_m - B_m)
        dif_v = np.absolute(A_v - B_v)

        return 1 - 0.5 * (dif_m.max() + dif_v.max())

    elif type == 'pk3':

        num = np.absolute(A_m - B_m) + np.absolute(A_v - B_v)
        denom = np.absolute(A_m + B_m) + np.absolute(A_v - B_v)
        D1 = np.sum(num)
        D2 = np.sum(denom)

        return 1 - (D1 / float(D2))

    elif type == 'new1':
        pe = np.absolute(A_m-B_m)
        ep = np.absolute(A_v - B_v)
        D = np.sum(pe + ep)


        return 1 - ((1 - np.exp(-0.5 * D)) / (1.0 - np.exp(-len(A))))
    elif type == 'new2':

        tt = np.absolute(np.sqrt(A_m) - np.sqrt(B_m))
        uu = np.absolute(np.sqrt(A_v ) - np.sqrt(B_v))
        D = np.sum(tt+uu)


        return 1 - ((1 - np.exp(-0.5 * D)) / float(1.0 - np.exp(-len(A))))
    else :
        print('Distance type parameter must be w1, w2, pk1, pk2, pk3, new1 or  new2 .')
        return

########################################################################################################################

def IFS_Sim12(A, B,w):
    '''
	IFS_SIM12: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by S.M. Chen, from the related article 
	"Measures of similarity between vague sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
    w = np.array(w)
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 or 1xN matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return


    sA = A_m - A_v
    sB = B_m - B_v
    D = np.sum(w * (np.subtract(1 , np.absolute(sA - sB)/2.0)))

    return D / float(np.sum(w))


########################################################################################################################

def IFS_Sim13(A,B ,p, type):
    '''
	IFS_SIM13: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung and M.S. Yang, from the related article 
	"Similarity measures of intuitionistic fuzzy sets based on Lp metric"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		type: Type of computed similarity: 'l', 'e' or 'c'
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if p < 1 and not isinstance(p, int):
        print('p parameter must be integer')
        return

    D = np.sum(np.power((np.absolute(np.power(A_m - B_m , p))+ np.power(np.absolute(A_v - B_v), p)),(1.0/p)))


    D = D / float(len(A))
    f1 = 2 ** (1.0 / float(p))

    if type == 'l':
        return (f1 - D) / float(f1)
    elif type == 'e':
        return (np.exp(-D) - np.exp(-f1)) / float(1 - np.exp(-f1))
    elif type == 'c':
        return (f1 - D) / float(f1 * (1.0 + D))

########################################################################################################################

def IFS_Sim14(A, B , w, a = 1,b = 0,c = 0):
    '''
	IFS_SIM14: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by D.H. Hong and C.Kim, from the related article 
	"A note on similarity measures between vague sets and between elements"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. Can be 0.
		a, b, c: Positive integers.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    n = len(A)

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return
    w = np.array(w)


    if w is None:
        w = np.full(A.size, 1 / float(A.size))

    elif (w.size == 1 and w != 0):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return

    if a < 0 or b < 0 or c < 0 :#or a > 1 or b > 1 or c > 1:
        print('Parameters ''a'', ''b'' and ''c'' must be equal or higher than 0')
        return

    if w.size == 1 and w == 0:#w != 0
        ##D = np.sum(np.divide(np.subtract(1 , (np.absolute(A_m - B_m) + np.absolute(A_v - B_v))),2.0))
        D = np.sum(np.subtract(1  , (np.absolute(A_m-B_m) + np.absolute(A_v-B_v))/2.0))
        return D / float(n)
    else:

        denom = a + b + c

        ptr1 = a * np.absolute(A_m - B_m)
        ptr2 = b * np.absolute(A_v - B_v)
        ptr3 = c * np.absolute(B_m + B_v - A_m - A_v)

        num = ptr1 + ptr2 + ptr3
        w = np.array(w)
        D = np.sum(w * (1 - num / float(denom)))#np.sum(np.multiply(w,np.divide(np.subtract(1,num),float(denom))))

        return D

########################################################################################################################

def IFS_Sim15(A, B,w, a = 1 , b = 0 , c = 0):
    '''
	IFS_SIM15: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by S.M. Chen, from the related article 
	"Similarity measure between vague sets and between elements"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. Can be 0.
		A,B,b,c: Positive integers.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if (a+b+c) != 1:
        print('Sum of parameters ''a'', ''b'' and ''c'' must be equal to 1.')
        return

    if a < c or c < 0 or 0 < b:
        print('a, b and c parameters must meet this condition: a >= c >= 0 >= b.')
        return

    if (a < 0 or a > 1) or (b < 0 or b > 1) or (c < 0 or c > 1):
        print('Parameters a, b and c must be 0 <= a, b, c <= 1.')
        return

    w = np.array(w)
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size == 1 and w != 0):
        print('Weight parameter dimensions must be 1-D Nx1 matrix.')
        return
    elif np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
        print('Weight values must be 0 <= w <= 1.')
        return

    elif (w.size != A.size or w.ndim != 1):
        print('Weight parameter dimensions must be 1-D Nx1 or 1xN matrix.')
        return



    denom = a - b

    ptr1 = a * (A_m - B_m)
    ptr2 = b* (A_v - B_v)
    ptr3 = c * (B_m + B_v -A_m-A_v)

    num = ptr1 + ptr2  + ptr3
    num = np.array(num)
    w = np.array(w)
    D = np.sum(w * np.absolute(num)/float(denom))
    D = np.array(D)

    return D / float(np.sum(w))

########################################################################################################################

def IFS_Sim16(A, B,p, w, a=1,b=0,c=0):
    '''
	IFS_SIM16: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by H.W. LA_m, from the related article 
	"New similarity measures between intuitionistic fuzzy sets and between elements"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		p: must be a positive integer.
		w: weights for the computed similarity. Can be 0.
		a,b,c: Positive integers. Must a + b + c = 1.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    n = len(A)

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[: , 2])

    if len(A[0]) != 3 or not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx3 matrices.')
        return

    w = np.array(w)

    if w is None:
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif w.size == 1 and w != 0:
        if w.size != A.size or w.ndim != 1:
            print('Weight parameter dimensions must be 1-D Nx1 matrix.')
            return
        if np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
            print('Weight values must be 0 <= w <= 1.')
            return
        if (a + b + c) != 1:
            print('Sum of parameters ''a'', ''b'' and ''c'' must be equal to 1.')
            return

    if p < 1 and not isinstance(p, int):
        print('p parameter must be integer')
        return

    if np.sum(w) == 0: #w.size == 1 and w != 0
        ptr1 = np.power(np.absolute(A_m - B_m), p)
        ptr2 = np.power(np.absolute(A_v - B_v), p)
        ptr3 = np.power(np.absolute(A_p - B_p), p)

        D = np.sum(ptr1 + ptr2 + ptr3)

        return 1 - (D / float(2.0 * n)) ** (1.0 / p)
    else:

        ptr1 = a * np.power(np.absolute(A_m - B_m),p)
        ptr2 = b * np.power(np.absolute(A_v - B_v), p)
        ptr3 = c * np.power(np.absolute(A_p - B_p), p)

        prts = ptr1 + ptr2 + ptr3
        D = np.sum(w * prts)

        return 1 - D ** (1.0 / p)

########################################################################################################################

def tOperator(a,b,lamda):
    a = np.array(a)
    b = np.array(b)
    if (lamda == np.inf):
        return np.maximum(0 , a + b - 1)
    elif (lamda == 0):
        return np.minimum(a,b)
    elif (lamda == 1):
        return a * b
    else:
        log_num = (np.power(lamda , np.subtract(1,a)) - 1 ) * (np.power(lamda , np.subtract(1,b)) - 1)
        log_denum = lamda - 1
        log_denum = np.array(log_denum)
        return np.log(1.0 + log_num / log_denum.astype(float))

########################################################################################################################

def IFS_Sim17(A,B, type, lamda = 1):
    '''
	IFS_SIM17: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by Ion Iancu, from the related article 
	"Intuitionistic fuzzy similarity measures based on Frank t-norms family"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		type: Type of computed similarity: 1-20.
		lambda: Frank family of t-operator parameter. Different cased of input:
		0, 1, Inf and other numbers.
	OUTPUT:
		S: Similarity measure.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    n = len(A)

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    if (type == 1):
        nominator = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / np.array(denom).astype("float")
    elif (type == 2):
        nominator = n - np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n - np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / np.array(denom).astype("float")
    elif (type == 3):
        nominator = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / (2.0 * n)
    elif (type == 4):
        nominator = n - np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / (2.0 * n)
    elif (type == 5):
        nom = n + np.sum(tOperator(A_m, B_m, lamda))
        denom = n + np.maximum(np.sum(A_m + A_v), np.sum(B_m+ B_v))
        return nom / np.array(denom).astype("float")
    elif (type == 6):
        nom = n + np.sum(tOperator(A_m, B_m, lamda))
        denom = n + np.maximum(np.sum(A_m + A_v), np.sum(B_m + B_v))
        return nom / np.array(denom).astype("float")
    elif (type == 7):
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_v - B_v)
        denom = n - np.sum(A_m + B_m - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / np.array(denom).astype("float")
    elif (type == 8):
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_m - B_m)
        denom = n - np.sum(A_v + B_v - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / np.array(np.sum(denom)).astype("float")
    elif (type == 9):
        nom = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.sum(A_m + B_m - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / float(denom)
    elif (type == 10):
        nom = n + np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.sum(A_v + B_v - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / float(denom)
    elif (type == 11):
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_v - B_v)
        denom = 2.0 * n
        return nom / denom
    elif (type == 12):
        nom = n - np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_m - B_m)
        denom = 2.0 * n
        return nom / denom
    elif (type == 13):
        nom = 2.0 * n + np.sum(2.0 * tOperator(A_m, B_m, lamda) + 2.0 * tOperator(A_v, B_v, lamda) - A_m - B_m - A_v - B_v)
        denom = 2.0 * n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - np.minimum(np.sum(A_m + A_v), np.sum(B_m + B_v)))
        return nom / denom
    elif (type == 14):
        nom = 2.0 * n + np.sum(2.0 * tOperator(A_m, B_m, lamda) + 2.0 * tOperator(A_v, B_v, lamda) - A_m - B_m - A_v - B_v)
        denom = 2.0 * n
        return nom / denom
    elif (type == 15):
        nom = np.minimum(np.sum(A_m + A_v), np.sum(B_m + B_v)) - np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda))
        denom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_m, lamda) - 2 * tOperator(A_v, B_v, lamda))
        return nom / denom
    elif (type == 16):
        nom = 2.0 * n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - np.maximum(np.sum(A_m + A_v), B_m + B_v))
        denom = 2.0 * n
        return nom / denom
    elif (type == 17):
        nom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_m, lamda) - 2.0 * tOperator(B_v, A_v, lamda))
        denom = n - np.sum(tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda) + np.maximum(A_m + B_v, A_v + A_v))
        return nom / denom.astype(float)
    elif (type == 18):
        nom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_v, lamda) - 2.0 * tOperator(B_m, A_v, lamda))
        denom = 2.0 * n
        return nom / denom
    elif (type == 19):
        nom = n + np.sum(tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda)) - np.maximum(np.sum(A_m + B_v), np.sum(B_m + A_v))
        denom = 2.0 * n + np.sum(2.0 * tOperator(A_m, B_v, lamda) + 2.0 * tOperator(B_m, A_v, lamda) - A_m - B_m - A_v - B_v)
        return nom / denom
    elif (type == 20):
        nom = n + np.minimum(np.sum(A_m + B_v), np.sum(B_m + A_v)) - np.sum(tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda))
        denom = 2.0 * n
        return nom / denom
    else:
        return None

########################################################################################################################

def IFS_Sim18(A,B, w):
    '''
	IFS_SIM18: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Yafei Song, Xiaodan Wang, Lei Lei, AiA_vun Xue, from the related article 
	"A novel similarity measure on intuitionistic fuzzy sets with its applications"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		w: weights for the computed similarity. Can be 0.
	OUTPUT:
		S: Similarity measure.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[: , 2])

    n = len(A)

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[: , 2])

    w = np.array(w)
	
    if w is None :
        w = np.full(A.shape[0] , 1 / float(A.shape[0]))
    elif (w.size == 1 and w != 0):
        if (w.size != A.size or w.ndim != 1):
            print('Weight parameter dimensions must be 1-D Nx1 matrix.')
            return
        if np.sum(w < 0) != 0 or np.sum(w > 1) != 0:
            print('Weight values must be 0 <= w <= 1.')
            return

    if w.size == 1 and w == 0:
        ff = 1.0 / (2.0 * n) * np.sum(
            np.sqrt(A_m * B_m) + np.multiply(2.0, np.sqrt(A_v, B_v)) + np.sqrt(A_p * B_p) + np.sqrt((1 - A_v) * (1 - B_v)))
        return ff
    else:
        lll = np.sum(w * (np.sqrt(A_m * B_m) + 2.0 * np.sqrt(A_v * B_v) + np.sqrt(A_p, B_p) + np.sqrt((1 - A_v) * (1 - B_v))))
        ff = 1.0 / 2.0 * lll
        return ff
		
########################################################################################################################

def IFS_Sim19(A , B):
    '''
	IFS_SIM19: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by Peerasak Intarapaiboon, from the related article 
	"A hierarchy-based similarity measure for intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity measure.
    '''
    n = len(A)
    A = np.array(A)
    A_m = np.array(A[: , 0])
    A_v = np.array(A[: , 1])

    B = np.array(B)
    B_m = np.array(B[: , 0])
    B_v = np.array(B[: , 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    ff = 1 - (1.0 / (2.0 / n) * np.sum(np.absolute(A_m - B_m) + np.absolute(A_v - B_v)))
    return ff
	
########################################################################################################################

def IFS_Sim20(A , B):
    '''
	IFS_SIM20: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Guannan Deng, Yanli A_viang, A_vingchao Fu, from the related article 
	"Monotonic similarity measures between intuitionistic fuzzy sets and their relationship with 
	entropy and inclusion measure"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity measure.
    '''
    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    gg = float(np.sum(np.minimum(A_m , B_m) + np.minimum(A_v , B_v)))/np.sum(np.maximum(A_m , B_m) + np.maximum(A_v , B_v))
    return gg
	
########################################################################################################################

def IFS_Sim21(A,B):
    '''
	IFS_SIM22: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Hoang Nguyen, from the related article 
	"A novel similarity/dissimilarity measure for intuitionistic fuzzy sets and its application
	in pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''
    n = len(A)
    A = np.array(A)
    A_m = A[:, 0]
    A_v = A[:, 1]
    A_p = A[:, 2]

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if len(A[0]) != 3 or not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx3 matrices.')
        return

    Ka = 1.0 / (n * np.sqrt(2)) * np.sum(np.sqrt(np.power(A_m, 2.0) + np.power(A_v, 2.0) + (1 - np.power(A_p, 2.0))))
    Kb = 1.0 / (n * np.sqrt(2)) * np.sum(np.sqrt(np.power(B_m, 2.0) + np.power(B_v, 2.0) + (1 - np.power(B_p, 2.0))))

    return 1 - np.absolute(Ka - Kb)
	
########################################################################################################################

def IFS_Sim22(A,B):
    '''
	IFS_SIM22: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Shyi-Ming Chen, Shou-HsA_mng Cheng,Tzu-Chun Lan, from the related article 
	"A novel similarity measure between intuitionistic fuzzy sets based on the centroid 
	points of transformed fuzzy numbers with applications to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''
    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if len(A[0]) != 3 or not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx3 matrices.')
        return

    term1 = np.absolute(2.0 * (A_m - B_m) - (A_v - B_v)) / 3.0 * (1 - ((A_p + B_p) / 2.0))
    term2 = np.absolute(2.0 * (A_v - B_v) - (A_m - B_m)) / 3.0 * ((A_p + B_p) / 2.0)

    return 1 - np.sum(term1) - np.sum(term2)
	
########################################################################################################################

def IFS_Sim23(A , B):
    '''
	IFS_SIM23: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by P. MuthukumarA,B, G. Sai Sundara KrishnanbA,B, from the related article 
	"A similarity measure of intuitionistic fuzzy soft sets and itsapplication in medical diagnosis"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B = np.array(B)
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (len(A[0]) != 2 or len(B[0]) != 2) and not (A.shape == B.shape):
        print ('A and B parameters must be 2-D Nx2 matrices.')
        return

    nom = np.sum(A_m * B_m + A_v * B_v)
    denom = np.sum(np.maximum(np.power(A_m, 2.0), np.power(B_m, 2.0)) + np.maximum(np.power(A_v, 2.0), np.power(B_v, 2.0)))
    return nom / float(denom)