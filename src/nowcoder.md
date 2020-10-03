[TOC]



#### 1.二维数组中的查找

**题目描述**
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数

<font color="red">题解1</font>：把每一行看成有序递增的数组，利用二分查找遍历每一行，时间复杂度是nlogn。

```cpp
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int m = array.size();
        for(int i=0; i<m; i++){
            int low = 0;
            int high = array[0].size()-1;
            while(low <= high){
                int mid = low+(high-low)/2;
                if(array[i][mid] == target){
                    return true;
                }else if(array[i][mid] > target){
                    high = mid-1;
                }else{
                    low = mid+1;
                }
            }
        }
        return false;
    }
};
```

<font color="red">题解2</font>:利用二维数组从上到下，从左到右递增的规律，选取右上角或左下角的元素开始与target比较。

```cpp
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        
        int row = array.size() - 1;
        int col = array[0].size() - 1;

        int i = row;
        int j = 0;
        while(i>=0&&j<=col){
            if(target < array[i][j]){
                i--;
            }else if(target > array[i][j]){
                j++;
            }else{
                return true;
            }
        }
        return false;
        return false;
    }
};
```



#### 2.替换空格

**题目描述**

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

```cpp
class Solution {
public:
	void replaceSpace(char *str,int length) {
        if(str == nullptr || length <= 0){
            return;
        }
        int count = 0;
        for(int i=0; i!=length; ++i){
            if(str[i] == ' '){
                ++count;
            }
        }
        if(count == 0){
            return;
        }
        int newlength = length + 2*count;
        for(int i=length; i>=0; --i){
            if(str[i] == ' '){
                str[newlength--] = '0';
                str[newlength--] = '2';
                str[newlength--] = '%';
            }else{
                str[newlength--] = str[i];
            }
        }
	}
};
```



#### 3.从尾到头打印链表

**题目描述**

输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

<font color="red">题解1</font>：非递归

```cpp
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> array;
        int count = 0;
        while(head){
            array.push_back(head->val);
            head = head->next;
            count++;
        }
        
        vector<int> result;
        for(int i=count-1; i>=0; i--){
            result.push_back(array[i]);
        }
        return result;
    }
};
```

<font color="red">题解2</font>：递归：if(!head)这个判定是必不可少的

```cpp
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> result;
        if(!head){
            return result;
        }
        result = printListFromTailToHead(head->next);
        result.push_back(head->val);
        return result;
    }
};
```



#### 4.重建二叉树

**题目描述**

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        TreeNode* res = reConstructBinaryTree(pre,0,pre.size()-1,vin,0,vin.size()-1);
        return res;
    }
    
private:
    TreeNode* reConstructBinaryTree(vector<int> pre, int startPre, int endPre, vector<int> vin, int startVin, int endVin){
        if(startPre>endPre||startVin>endVin){
            return NULL;
        }
        TreeNode* root = new TreeNode(pre[startPre]);
        for(int i=startVin; i<=endVin; i++){
            if(vin[i]==pre[startPre]){
                root->left = reConstructBinaryTree(pre, startPre+1, startPre+i-startVin, vin, startVin, i-1);
                root->right = reConstructBinaryTree(pre, startPre+i-startVin+1, endPre, vin, i+1, endVin);
                break;
            }
        }
        return root;
    }
};
```

```cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {

        if(vin.size() == 0){
            return NULL;
        }
        vector<int> left_pre, right_pre, left_vin, right_vin;
        TreeNode* head = new TreeNode(pre[0]);
        int index = 0;
        for(int i=0; i<vin.size(); i++){
            if(vin[i] == pre[0]){
                index = i;
                break;
            }
        }
        
        for(int i=0; i<index; i++){
            left_vin.push_back(vin[i]);
            left_pre.push_back(pre[i+1]);
        }
        
        for(int i=index+1; i<vin.size(); i++){
            right_vin.push_back(vin[i]);
            right_pre.push_back(pre[i]);
        }
        head->left = reConstructBinaryTree(left_pre, left_vin);
        head->right = reConstructBinaryTree(right_pre, right_vin);
        return head;
    }
};
```



#### 5.用两个栈实现队列

**题目描述**

```cpp
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.empty()){
            while(!stack1.empty()){
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        int t = stack2.top();
        stack2.pop();
        return t;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```



#### 6.旋转数组的最小数字

**题目描述**

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```cpp
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int left = 0;
        int right = rotateArray.size()-1;
        while(left < right){
            int mid = left+(right-left)/2;
            if(rotateArray[mid]>rotateArray[right]){
                left = mid+1;
            }else if(rotateArray[mid]==rotateArray[right]){
                right = right-1;
            }else{
                right = mid;
            }
        }
        return rotateArray[left];
    }
};
```

注：当rotateArray[mid]>rotateArray[right]时，要让left=mid+1；如果在最前面判断if(right-left==1){mid = left;break;}时，则不需要加1.

```cpp
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int low = 0;
        int high = rotateArray.size()-1;
        int mid=-1;
        while(low<high){
            if(high-low==1){
                mid=high;
                break;
            }
            mid = low+(high-low)/2;
            if(rotateArray[low]==rotateArray[high]&&rotateArray[mid]==rotateArray[low]){
                return minOrder(rotateArray, low, high);
            }
            if(rotateArray[mid]>=rotateArray[low]){
                low = mid;
            }else{
                high=mid;
            }
        }
        return rotateArray[mid];
    }
private:
    int minOrder(vector<int> &num, int low, int high){
        int res = num[low];
        for(int i=low+1;i<high;i++){
            if(num[i]<res){
                res=num[i];
            }
        }
        return res;
    }
};
```



#### 7.斐波那契数列

**题目描述**

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

n<=39

```java
class Solution {
public:
    int Fibonacci(int n) {

        if(n==0){
            return 0;
        }
        if(n==1){
            return 1;
        }
        int left=0;
        int right=1;
        for(int i=2; i<=n; i++){
            int tmp = right;
            right=left+right;
            left=tmp;
        }
        return right;
    }
};
```



#### 8.跳台阶

**题目描述**

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）

```cpp
class Solution {
public:
    int jumpFloor(int number) {
        if(number<=0){
            return -1;
        }
        if(number==1){
            return 1;
        }
        if(number==2){
            return 2;
        }
        return jumpFloor(number-1)+jumpFloor(number-2);
    }
};
```



```cpp
class Solution {
public:
    int jumpFloor(int number) {
        if(number<=0){
            return -1;
        }
        if(number==1){
            return 1;
        }
        if(number==2){
            return 2;
        }
        int left=1;
        int right=2;
        for(int i=3;i<=number;i++){
            right+=left;
            left=right-left;
        }
        return right;
    }
};
```



#### 9.变态跳台阶

**题目描述**

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```cpp
class Solution {
public:
    int jumpFloorII(int number) {

        if(number <= 0){
            return -1;
        }else if(number == 1){
            return 1;
        }else{
            return 2*jumpFloorII(number-1);
        }
    }
};
```



#### 10.矩阵覆盖

**题目描述**

```cpp
// 递归
class Solution {
public:
    int rectCover(int number) {

        if(number<=0){
            return 0;
        }
        if(number==1){
            return 1;
        }else if(number==2){
            return 2;
        }else{
            return rectCover(number-1)+rectCover(number-2);
        }
    }
};
```



#### 11.二进制中1的个数

**题目描述**

输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。

```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int mask = 0x01;
         int count = 0;
         while(mask!=0){
             if(mask&n){
                 count++;
             }
             mask<<=1;
         }
         return count;
     }
};
```



```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int count = 0;
         while(n!=0){
             n=n&(n-1);
             count++;
         }
         return count;
     }
};
```



#### 12.数值的整数次方

**题目描述**

给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

保证base和exponent不同时为0

```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int count = 0;
         while(n!=0){
             n=n&(n-1);
             count++;
         }
         return count;
     }
};
```



```cpp
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent < 0){
            base = 1/base;
            exponent = -exponent;
        }
        return cen_Power(base, exponent);
    }
private:
    double cen_Power(double base, int exponent){
        if(exponent == 0){
            return 1.0;
        }
        double ret = cen_Power(base, exponent/2);
        if(exponent % 2 == 1){
            return ret*ret*base;
        }else{
            return ret*ret;
        }
    }
};
```



```cpp
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent < 0){
            exponent = -exponent;
            base = 1/base;
        }
        double p = base;
        double res = 1.0;
        while(exponent){
            if(exponent&1){
                res *= p;
            }
            p*=p;
            exponent>>=1;
        }
        return res;
    }
};
```



#### 13.调整数组顺序使奇数位于偶数前面

**题目描述**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

```cpp
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int i=0;
        int j;
        while(i<array.size()){
            while(i<array.size()&&array[i]%2==1){
                i++;
            }
            j=i+1;
            while(j<array.size()&&array[j]%2==0){
                j++;
            }
            if(j<array.size()){
                int temp = array[j];
                for(int k=j-1; k>=i; k--){
                    array[k+1] = array[k];
                }
                array[i++] = temp;
            }else{
                break;
            }
        }
    }
};
```



```cpp
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int i = 0;
        for(int j=0; j<array.size(); j++){
            if(array[j]%2==1){
                int temp = array[j];
                for(int k=j-1; k>=i; k--){
                    array[k+1] = array[k];
                }
                array[i++] = temp;
            }
        }
    }
};
```



#### 14.链表中倒数第k个结点

**题目描述**

输入一个链表，输出该链表中倒数第k个结点。

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
    
        if(k<=0 || !pListHead){
            return NULL;
        }
        int n = 0;
        ListNode* cur = pListHead;
        while(cur){
            n++;
            cur = cur->next;
        }
        if(n < k){
            return NULL;
        }
        n-=k;
        while(n--){
            pListHead = pListHead->next;
        }
        return pListHead;
    }
};
```



```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(k<=0||!pListHead){
            return NULL;
        }
        ListNode* fast = pListHead;
        ListNode* slow = pListHead;
        while(k--){
            if(fast){
                fast = fast->next;
            }
            else{
                return NULL;
            }
        }
        while(fast){
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```



#### 15.反转链表

**题目描述**

输入一个链表，反转链表后，输出新链表的表头。

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {

        ListNode* cur = pHead;
        ListNode* pre = NULL;
        ListNode* nex = NULL;
        while(cur!=NULL){
            nex = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nex;
        }
        return pre;
    }
};
```



#### 16.合并两个排序的链表

**题目描述**

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        ListNode* res = new ListNode(-1);
        res->next = NULL;
        ListNode* cur = res;
        while(p1!=NULL&&p2!=NULL){
            if(p1->val<p2->val){
                res->next = p1;
                res = p1;
                p1 = p1->next;
            }else{
                res->next = p2;
                res = p2;
                p2 = p2->next;
            }
        }
        if(p1!=NULL){
            res->next=p1; 
        }
        if(p2!=NULL){
            res->next=p2;
        }
        return cur->next;
    }
};
```



```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == NULL){
            return pHead2;
        }
        if(pHead2 == NULL){
            return pHead1;
        }
        if(pHead1->val < pHead2->val){
            pHead1->next = Merge(pHead1->next, pHead2);
            return pHead1;
        }else{
            pHead2->next = Merge(pHead1, pHead2->next);
            return pHead2;
        }
    }
};
```



#### 17.树的子结构

**题目描述**

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1==NULL||pRoot2==NULL){
            return false;
        }
        return isSubtree(pRoot1, pRoot2)||HasSubtree(pRoot1->left, pRoot2)||HasSubtree(pRoot1->right, pRoot2);
    }
private:
    bool isSubtree(TreeNode* pRootA, TreeNode* pRootB){
        if(pRootB == NULL){
            return true;
        }
        if(pRootA == NULL){
            return false;
        }
        if(pRootB->val == pRootA->val){
            return isSubtree(pRootA->left, pRootB->left)&&isSubtree(pRootA->right, pRootB->right);
        }else{
            return false;
        }
    }
};
```



#### 18.二叉树的镜像

**题目描述**

操作给定的二叉树，将其变换为源二叉树的镜像。

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(pRoot == NULL){
            return;
        }
        TreeNode* tmp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = tmp;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};
```



#### 19.顺时针打印矩阵

**题目描述**

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```cpp
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {

        vector<int> res;
        if(matrix.size()==0||matrix[0].size()==0){
            return res;
        }
        int up = 0;
        int down = matrix.size()-1;
        int left = 0;
        int right = matrix[0].size()-1;
        while(true){
            for(int col=left; col<=right; col++){
                res.push_back(matrix[up][col]);
            }
            up++;
            if(up > down){
                break;
            }
            for(int row=up; row<=down; row++){
                res.push_back(matrix[row][right]);
            }
            right--;
            if(left > right){
                break;
            }
            for(int col=right; col>=left; col--){
                res.push_back(matrix[down][col]);
            }
            down--;
            if(up > down){
                break;
            }
            for(int row=down; row>=up; row--){
                res.push_back(matrix[row][left]);
            }
            left++;
            if(left > right){
                break;
            }
        }
        return res;
    }
};
```



```cpp
class Solution {
public:
    vector<int> res;
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int tR = 0;
        int tC = 0;
        int dR = matrix.size()-1;
        int dC = matrix[0].size()-1;
        while(tR<=dR&&tC<=dC){
            printEdge(matrix, tR++, tC++, dR--, dC--);
        }
        return res;
    }
    void printEdge(vector<vector<int> > matrix, int tR, int tC, int dR, int dC){
        if(tR==dR){
            for(int i=tC; i<=dC; i++){
                res.push_back(matrix[tR][i]);
            }
        }
        else if(tC==dC){
            for(int i=tR; i<=dR; i++){
                res.push_back(matrix[i][tC]);
            }
        }
        else{
            int curC = tC;
            int curR = tR;
            while(curC != dC){
                res.push_back(matrix[tR][curC]);
                curC++;
            }
            while(curR != dR){
                res.push_back(matrix[curR][dC]);
                curR++;
            }
            while(curC != tC){
                res.push_back(matrix[dR][curC]);
                curC--;
            }
            while(curR != tR){
                res.push_back(matrix[curR][tC]);
                curR--;
            }
        }
    }
};
```



#### 20.包含min函数的栈

**题目描述**

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

```cpp
class Solution {
public:
    stack<int> stack1, stack2;
    void push(int value) {
        stack1.push(value);
        if(stack2.empty()){
            stack2.push(value);
        }else if(value <= stack2.top()){
            stack2.push(value);
        }
    }
    void pop() {
        if(stack1.top()==stack2.top()){
            stack2.pop();
        }
        stack1.pop();
    }
    int top() {
        return stack1.top();
    }
    int min() {
        return stack2.top();
    }
};
```



#### 21.栈的压入弹出序列

**题目描述**

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

```cpp
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        vector<int> stack;
        if(pushV.size()==0){
            return false;
        }
        for(int i=0,j=0; i<pushV.size();){
            stack.push_back(pushV[i++]);
            while(j<popV.size()&&stack.back()==popV[j]){
                stack.pop_back();
                j++;
            }
        }
        return stack.empty();
    }
};
```



#### 22.从上往下打印二叉树

**题目描述**

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {

        vector<int> res;
        queue<TreeNode*> nodes;
        if(root == NULL){
            return res;
        }
        nodes.push(root);
        res.push_back(root->val);
        while(!nodes.empty()){
            TreeNode* node = nodes.front();
            nodes.pop();
            if(node->left!=NULL){
                nodes.push(node->left);
                res.push_back(node->left->val);
            }
            if(node->right!=NULL){
                nodes.push(node->right);
                res.push_back(node->right->val);
            }
        }
        return res;
        
    }
};
```



```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {

        vector<int> res;
        if(!root){
            return res;
        }
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            TreeNode* node = q.front();
            q.pop();
            res.push_back(node->val);
            if(node->left){
                q.push(node->left);
            }
            if(node->right){
                q.push(node->right);
            }
        }
        return res;
    }
};
```



#### 23.二叉搜索树的后序遍历序列

**题目描述**

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

```cpp
class Solution {
    bool judge(vector<int> &a, int l, int r){
        if(l>=r){
            return true;
        }
        int key = a[r];
        int i;
        for(i=l; i<r; i++){
            if(a[i] > key){
                break;
            }
        }
        for(int j=i; j<r; j++){
            if(a[j]<key){
                return false;
            }
        }
        return judge(a, l, i-1)&&judge(a, i, r-1);
    }
public:
    bool VerifySquenceOfBST(vector<int> sequence) {

        if(sequence.size()==0){
            return false;
        }
        return judge(sequence, 0, sequence.size()-1);
    }
};
```



#### 24.二叉树中和为某一值的路径

**问题分析**

输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
    
public:
    vector<vector<int>> res;
    vector<int> ret;
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        
        if(root==NULL){
            return res;
        }
        ret.push_back(root->val);
        expectNumber-=root->val;
        if(expectNumber==0&&root->left==NULL&&root->right==NULL){
            res.push_back(ret);
        }
        FindPath(root->left, expectNumber);
        FindPath(root->right, expectNumber);
        ret.pop_back();
        return res;
    }
};
```



#### 25.复杂链表的复制

**题目描述**

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

```cpp
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead==NULL){
            return NULL;
        }
        RandomListNode* currentNode = pHead;
        while(currentNode != NULL){
            RandomListNode* cloneNode = new RandomListNode(currentNode->label);
            RandomListNode* nextNode = currentNode->next;
            currentNode->next = cloneNode;
            cloneNode->next = nextNode;
            currentNode = nextNode;
        }
        
        currentNode = pHead;
        while(currentNode != NULL){
            currentNode->next->random = currentNode->random==NULL?NULL:currentNode->random->next;
            currentNode = currentNode->next->next;
        }
        
        currentNode = pHead;
        RandomListNode* pCloneHead = pHead->next;
        while(currentNode != NULL){
            RandomListNode* cloneNode = currentNode->next;
            currentNode->next = cloneNode->next;
            cloneNode->next = cloneNode->next==NULL?NULL:cloneNode->next->next;
            currentNode = currentNode->next;
        }
        return pCloneHead;
    }
};
```



#### 26.二叉搜索树与双向链表

**题目描述**

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == NULL){
            return NULL;
        }
        if(pRootOfTree->left==NULL&&pRootOfTree->right==NULL){
            return pRootOfTree;
        }
        
        TreeNode* left = Convert(pRootOfTree->left);
        TreeNode* p = left;
        while(p!=NULL&&p->right!=NULL){
            p = p->right;
        }
        if(left != NULL){
            p->right = pRootOfTree;
            pRootOfTree->left = p;
        }
        
        TreeNode* right = Convert(pRootOfTree->right);
        if(right != NULL){
            right->left = pRootOfTree;
            pRootOfTree->right = right;
        }
        
        return left!=NULL?left:pRootOfTree;
    }
};
```



#### 27.字符串的排列

**题目描述**

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

```
输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
```

```cpp
class Solution {
public:
    vector<string> Permutation(string str) {
        vector<string> result;
        if(str.empty()){
            return result;
        }
        Permutation(str, result, 0);
        sort(result.begin(), result.end());
        return result;
    }
    
    void Permutation(string str, vector<string> &result, int begin){
        if(begin == str.size()-1){
            if(find(result.begin(), result.end(), str)==result.end()){
                result.push_back(str);
            }
        }else{
            for(int i=begin; i<str.size(); i++){
                swap(str[i], str[begin]);
                Permutation(str, result, begin+1);
                swap(str[i], str[begin]);
            }
        }
    }
    void swap(char &fir, char &sec){
        char temp = fir;
        fir = sec;
        sec = temp;
    }
};
```



#### 28.数组中出现次数超过一半的数字

**题目描述**

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

```cpp
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        map<int, int> m;
        int len = numbers.size();
        int count;
        for(int i=0; i<len; i++){
            count = ++m[numbers[i]];
            if(count > len/2){
                return numbers[i];
            }
        }
        return 0;
    }
};
```

#### 29.最小的k个数

**题目描述**

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

```cpp
//优先队列
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> res;
        int length = input.size();
        if(k>length||input.empty()){
            return res;
        }
        priority_queue<int, greater<int>> a;
        for(int i=0; i<length; i++){
            a.push(input[i]);
        }
        for(int i=0; i<k; i++){
            res.push_back(a.top());
            a.pop();
        }
        return res;
    }
};
```



#### 30.连续子数组最大和

**题目描述**

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

```cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(array.size()==0){
            return 0;
        }
        int result = array[0];
        int res = result;
        for(int i=1; i<array.size(); i++){
            result += array[i];
            if(result < 0){
                result=array[i+1];
                i++;
            }
            if(result > res){
                res = result;
            }
        }
        return res;
    }
};
```



```cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(!array.size())
            return 0;
        int mx=INT_MIN;
        for(int i=0,s=0;i<array.size();i++){
            s=max(s+array[i],array[i]);
            mx=max(mx,s);
        }
        return mx;
    }
};
```

#### 31.数组中1出现的次数

**题目描述**

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

```cpp
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        if(n <= 0){
            return 0;
        }
        int count = 0;
        for(int i=1; i<=n; i*=10){
            int diver = i*10;
            count += (n/diver)*i + min(max(0, n%diver-i+1), i);
        }
        return count;
    }
};
```

#### 32.把数组排成最小的数

**题目描述**

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```cpp
class Solution {
public:
    static bool cmp(int a, int b){
        string A = "";
        string B = "";
        A+=to_string(a);
        A+=to_string(b);
        B+=to_string(b);
        B+=to_string(a);
        return A < B;
    }
    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(), numbers.end(), cmp);
        string res = "";
        for(int i=0; i<numbers.size(); i++){
            res += to_string(numbers[i]);
        }
        return res;
    }
};
```

#### 33.丑数

**题目描述**

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

```cpp
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if (index < 7)return index;
        vector<int> res(index);
        res[0] = 1;
        int t2=0, t3=0, t5=0;
        for(int i=1; i<index; i++){
            res[i]=min(res[t2]*2, min(res[t3]*3, res[t5]*5));
            if(res[i] == res[t2]*2) t2++;
            if(res[i] == res[t3]*3) t3++;
            if(res[i] == res[t5]*5) t5++;
        }
        return res[index-1];
    }
};
```

#### 34.第一个只出现一次的字符

**题目描述**

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

```cpp
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        map<char, int> mp;
        for(int i=0; i<str.size(); i++){
            mp[str[i]]++;
        }
        for(int i=0; i<str.size(); i++){
            if(mp[str[i]] == 1){
                return i;
            }
        }
        return -1;
    }
};
```

#### --35.数组中的逆序对

**题目描述**

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007



#### 36.两个链表的第一个公共节点

**题目描述**

输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int len1 = findListLength(pHead1);
        int len2 = findListLength(pHead2);
        if(len1 > len2){
            pHead1 = walkStep(pHead1, len1-len2);
        }else{
            pHead2 = walkStep(pHead2, len2-len1);
        }
        while(pHead1 != NULL){
            if(pHead1 == pHead2){
                return pHead1;
            }
            pHead1 = pHead1->next;
            pHead2 = pHead2->next;
        }
        return NULL;
    }
    int findListLength(ListNode* pHead){
        if(pHead == NULL){
            return 0;
        }
        int sum=1;
        while(pHead = pHead->next){
            sum++;
        }
        return sum;
    }
    ListNode* walkStep(ListNode* pHead, int step){
        while(step--){
            pHead = pHead->next;
        }
        return pHead;
    }
};
```

#### 37.数字在升序数组中出现的次数

**题目描述**

统计一个数字在升序数组中出现的次数。

```cpp
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int count=0;
        int len = data.size();
        int left = 0;
        int right = len-1;
        while(right >= left){
            int mid = left+(right-left)/2;
            if(data[mid]>k){
                right=mid-1;
            }else if(data[mid]<k){
                left=mid+1;
            }else{
                count++;
                int re = mid;
                while(data[--mid]==k) count++;
                while(data[++re]==k) count++;
                return count;
            }
        }
        return count;
    }
};
```

#### 38.二叉树的深度

**题目描述**

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == NULL){
            return 0;
        }
        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        return max(left, right)+1;
    }
};
```

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == NULL){
            return 0;
        }
        queue<TreeNode*> queue;
        queue.push(pRoot);
        int depth=0, count=0, nextCount=1;
        while(!queue.empty()){
            TreeNode* top = queue.front();
            count++;
            queue.pop();
            if(top->left != NULL){
                queue.push(top->left);
            }
            if(top->right != NULL){
                queue.push(top->right);
            }
            if(count == nextCount){
                nextCount = queue.size();
                count = 0;
                depth++;
            }
        }
        return depth;
    }
};
```

#### 39.平衡二叉树

**题目描述**

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树

```cpp
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(pRoot == NULL){
            return true;
        }
        return abs(depth(pRoot->left)-depth(pRoot->right))<=1&&IsBalanced_Solution(pRoot->left)&&IsBalanced_Solution(pRoot->right);
    }
    int depth(TreeNode* pRoot){
        if(pRoot == NULL){
            return 0;
        }
        int left = depth(pRoot->left);
        int right = depth(pRoot->right);
        return max(left, right)+1;
    }
};
```

#### 40.数组中只出现一次的数字

**题目描述**

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

```cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        map<int, int> mp;
        vector<int> res;
        for(int i=0; i<data.size(); i++){
            mp[data[i]]++;
        }
        for(int i=0; i<data.size(); i++){
            if(mp[data[i]]==1){
                res.push_back(data[i]);
            }
        }
        *num1 = res[0];
        *num2 = res[1];
    }
};
```

```cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        if(data.size()<2){
            return ;
        }
        int bitResult=0;
        for(int i=0; i<data.size(); i++){
            bitResult^=data[i];
        }
        int index = 0;
        while((bitResult&1)==0){
            bitResult>>=1;
            index++;
        }
        *num1=*num2=0;
        for(int i=0; i<data.size(); i++){
            if(IsBit(data[i], index)){
                *num1^=data[i];
            }else{
                *num2^=data[i];
            }
        }
    }
    bool IsBit(int num, int indexBit){
        num = num>>indexBit;
        return (num&1)==1;
    }
};
```

#### 41.和为S的连续正数序列

**题目描述**

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!。输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。

```cpp
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> result;
        int plow = 1, phigh = 2;
        while(phigh>plow){
            int cur = (phigh+plow)*(phigh-plow+1)/2;
            if(cur == sum){
                vector<int> res;
                for(int i=plow; i<=phigh; i++){
                    res.push_back(i);
                }
                result.push_back(res);
                plow++;
            }else if(cur < sum){
                phigh++;
            }else{
                plow++;
            }
        }
        return result;
    }
};
```

#### 42.和为S的两个数字

**题目描述**

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

```cpp
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int> result;
        int len = array.size();
        int plow = 0;
        int phigh = len-1;
        while(plow<phigh){
            if(array[plow]+array[phigh]==sum){
                result.push_back(array[plow]);
                result.push_back(array[phigh]);
                break;
            }
            while(plow<phigh&&array[plow]+array[phigh]>sum) phigh--;
            while(plow<phigh&&array[plow]+array[phigh]<sum) plow++;
        }
        return result;
    }
};
```

#### 43.左旋转字符串

**题目描述**

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

```cpp
class Solution {
public:
    string LeftRotateString(string str, int n) {
        string res;
        if(str.size()>0&&str.size()>=n&&n>=0){
            res += str.substr(n);
            res += str.substr(0, n); 
        }
        return res;
    }
};
```

#### 44.反转单词顺序列

**题目描述**

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

```cpp
class Solution {
public:
    string ReverseSentence(string str) {
        string res = "";
        string temp = "";
        for(int i=0; i<str.size(); i++){
            if(str[i] == ' '){
                res = ' '+temp+res;;
                temp = "";
            }else{
                temp += str[i];
            }
        }
        if(temp.size()){
            res = temp + res;
        }
        return res;
    }
};
```

#### 45.扑克牌顺子

**题目描述**

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

```cpp
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        int len = numbers.size();
        if(len != 5){
            return false;
        }
        sort(numbers.begin(), numbers.end());
        int jokers = 0;
        for(int i=0; i<5&&numbers[i]==0; i++){
            jokers++;
        }
        if(jokers > 4){
            return false;
        }
        for(int i=jokers+1; i<5; i++){
            if(numbers[i] == numbers[i-1]){
                return false;
            }
        }
        int dis = numbers[4]-numbers[jokers];
        if(dis <= 4){
            return true;
        }
        return false;
    }
};
```

#### 46.圆圈中最后剩下的数

**题目描述**

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

如果没有小朋友，请返回-1

```cpp
class Solution {
public:
    int LastRemaining_Solution(int n, int m)
    {
        if(n<1||m<1) return -1;
        int array[n];
        for(int i=0; i<n; i++){
            array[i] = 0;
        }
        int i=-1, step=0, count=n;
        while(count>0){
            i++;
            if(i>=n) i=0;
            if(array[i] == -1) continue;
            step++;
            if(step == m){
                array[i]=-1;
                step = 0;
                count--;
            }
        }
        return i;
    }
};
```

#### 47.求1+2+3+...+n

**题目描述**

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

```cpp
class Solution {
public:
    int Sum_Solution(int n) {
        int sum = n;
        n&&(sum+=Sum_Solution(n-1));
        return sum;
    }
};
```

#### 48.不用加减乘除做加法

**题目描述**

写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

```cpp
class Solution {
public:
    int Add(int num1, int num2)
    {
        while(num2 != 0){
            int temp = num1^num2;
            num2 = (num1&num2)<<1;
            num1 = temp;
        }
        return num1;
    }
};
```

#### 49.把字符串转换成整数

**题目描述**

将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

```cpp
class Solution {
public:
    int StrToInt(string str) {
        int sum = 0;
        if(str.size() == 0){
            return 0;
        }
        int s = 1;
        if(str[0]=='-') s = -1;
        if(str[0]=='+') s = 1;
        for(int i=(str[0]=='-'||str[0]=='+')?1:0; i<str.size(); i++){
            if(str[i]<'0'||str[i]>'9'){
                return 0;
            }
            int temp = str[i]-'0';
            sum = 10*sum+temp;
        }
        return s*sum;
    }
};
```

#### 50.数组中重复的数字

**题目描述**

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

```cpp
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        if(numbers==nullptr||length<=0){
            return false;
        }
        for(int i=0; i<length; i++){
            if(numbers[i]<0||numbers[i]>length-1){
                return false;
            }
        }
        for(int i=0; i<length; i++){
            while(numbers[i]!=i){
                if(numbers[i]==numbers[numbers[i]]){
                    *duplication = numbers[i];
                    return true;
                }
                int temp = numbers[i];
                numbers[i] = numbers[temp];
                numbers[temp] = temp;
            }
        }
        return false;
    }
};
```

#### 51.构建乘积数组

**题目描述**

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）

对于A长度为1的情况，B无意义，故而无法构建，因此该情况不会存在。

```cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> B(len, 0);
        if(len == 1){
            return B;
        }
        for(int i=0; i<len; i++){
            int s = 1;
            for(int j=0; j<len; j++){
                if(j == i){
                    continue;
                }
                s*=A[j];
            }
            B[i] = s;
        }
        return B;
    }
};
```

```cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> B(len, 1);
        if(len <= 1){
            return B;
        }
        for(int i=1; i<len; i++){
            B[i] = B[i-1]*A[i-1];
        }
        int temp = 1;
        for(int i=len-2; i>=0; i--){
            temp *= A[i+1];
            B[i] = B[i]*temp;
        }
        return B;
    }
};
```

#### 52.正则表达式匹配

**题目描述**

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

```cpp
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if(*str=='\0'&&*pattern=='\0')
            return true;
        if(*str!='\0'&&*pattern=='\0')
            return false;
        if(*(pattern+1)!='*'){
            if(*str==*pattern||(*str!='\0'&&*pattern=='.'))
                return match(str+1,pattern+1);
            else{
                return false;
            }
        }else{
            if(*pattern==*str||(*str!='\0'&&*pattern=='.'))
                return match(str+1,pattern+2)||match(str+1,pattern)||match(str,pattern+2);
            else
                return match(str,pattern+2);
        }
    }
};
```

#### 53.表示数值的字符串

**题目描述**

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

```cpp
class Solution {
public:
    bool isNumeric(char* str)
    {
        bool sign=false,decimal=false,hasE=false;
        for(int i=0;i<strlen(str);i++){
            if(str[i]=='e'||str[i]=='E'){
                if(i==strlen(str)-1)
                    return false;
                if(hasE) return false;
                hasE=true;
            }else if(str[i]=='+'||str[i]=='-'){
                if(sign&&str[i-1]!='e'&&str[i-1]!='E')
                    return false;
                if(!sign&&i>0&&str[i-1]!='e'&&str[i-1]!='E')
                    return false;
                sign=true;
            }else if(str[i]=='.'){
                if(hasE||decimal)
                    return false;
                decimal=true;
            }else if(str[i]<'0'||str[i]>'9')
                return false;
        }
        
        return true;
    }

};
```

```cpp
class Solution {
public:
    bool isNumeric(char* string)
    {
        if(string == nullptr){
            return false;
        }
        bool numeric = scanInteger(&string);
        if(*string=='.'){
            string++;
            numeric = scanUnsighedInteger(&string)||numeric;
        }
        if(*string=='e'||*string=='E'){
            string++;
            numeric = numeric&&scanInteger(&string);
        }
        return numeric&&*string=='\0';
    }
    bool scanInteger(char** str){
        if(**str=='+'||**str=='-'){
            ++(*str);
        }
        return scanUnsighedInteger(str);
    }
    bool scanUnsighedInteger(char** str){
        char* before = *str;
        while(**str!='\0'&&(**str>='0')&&(**str<='9')){
            ++(*str);
        }
        return *(str)>before;
    }

};
```

#### 54.字符流中第一个不重复的字符

**题目描述**

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

```cpp
class Solution
{
public:
  //Insert one char from stringstream
    string s;
    char hash[256] = {0};
    void Insert(char ch)
    {
         s += ch;
        hash[ch]++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        int size = s.size();
        for(int i=0; i<size; i++){
            if(hash[s[i]] == 1){
                return s[i];
            }
        }
        return '#';
    }
    
};
```

#### 55.链表中环的入口节点

**题目描述**

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead==NULL||pHead->next==NULL){
            return NULL;
        }
        ListNode* p1 = pHead;
        ListNode* p2 = pHead;
        while(p2!=NULL&&p2->next!=NULL){
            p1 = p1->next;
            p2 = p2->next->next;
            if(p1 == p2){
                p2 = pHead;
                while(p1 != p2){
                    p1 = p1->next;
                    p2 = p2->next;
                }
                return p1;
            }
        }
        return NULL;
    }
};
```

#### 56.删除链表中重复的节点

**题目描述**

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead == NULL || pHead->next == NULL){
            return pHead;
        }
        if(pHead->val == pHead->next->val){
            ListNode* pNode = pHead->next;
            while(pNode!=NULL&&pNode->val==pHead->val){
                pNode = pNode->next;
            }
            return deleteDuplication(pNode);
        }else{
            pHead->next = deleteDuplication(pHead->next);
            return pHead;
        }
    }
};
```

#### 57.二叉树的下一个节点

**题目描述**

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

```cpp
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if(pNode==NULL){
            return NULL;
        }
        if(pNode->right!=NULL){
            pNode=pNode->right;
            while(pNode->left!=NULL){
                pNode=pNode->left;
            }
            return pNode;
        }
        while(pNode->next!=NULL){
            if(pNode->next->left==pNode){
                return pNode->next;
            }
            pNode=pNode->next;
        }
        return NULL;
    }
};
```

#### 58.对称的二叉树

**题目描述**

请实现一个函数，用来判断一棵二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot==NULL){
            return true;
        }
        return combat(pRoot->left, pRoot->right);
    }
    bool combat(TreeNode* left, TreeNode* right){
        if(left==NULL){
            if(right==NULL){
                return true;
            }
            return false;
        }
        if(right==NULL){
            return false;
        }
        if(left->val!=right->val){
            return false;
        }
        return combat(left->left, right->right)&&combat(left->right, right->left);
    }
};
```

#### 59.按之字形顺序打印二叉树

**题目描述**

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int>> result;
        stack<TreeNode*> stack1, stack2;
        if(pRoot==NULL){
            return result;
        }
        stack1.push(pRoot);
        while(!stack1.empty()||!stack2.empty()){
            vector<int> ret1, ret2;
            TreeNode* cur = NULL;
            while(!stack1.empty()){
                cur=stack1.top();
                if(cur->left){
                    stack2.push(cur->left);
                }
                if(cur->right){
                    stack2.push(cur->right);
                }
                ret1.push_back(stack1.top()->val);
                stack1.pop();
            }
            if(ret1.size()){
                result.push_back(ret1);
            }
            while(!stack2.empty()){
                cur = stack2.top();
                if(cur->right){
                    stack1.push(cur->right);
                }
                if(cur->left){
                    stack1.push(cur->left);
                }
                ret2.push_back(stack2.top()->val);
                stack2.pop();
            }
            if(ret2.size()){
                result.push_back(ret2);
            }       
        }
        return result;
    }
    
};
```

#### 60.把二叉树打印成多行

**题目描述**

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int>> result;
            queue<TreeNode*> queue1;
            if(pRoot==NULL){
                return result;
            }
            queue1.push(pRoot);
            while(!queue1.empty()){
                int lo=0, hi=queue1.size()-1;
                vector<int> ret;
                while(lo++<=hi){
                    TreeNode* cur=queue1.front();
                    queue1.pop();
                    if(cur->left){
                        queue1.push(cur->left);
                    }
                    if(cur->right){
                        queue1.push(cur->right);
                    }
                    ret.push_back(cur->val);
                }
                result.push_back(ret);
            }
            return result;
        }
};
```

#### --61.序列化二叉树

**题目描述**

请实现两个函数，分别用来序列化和反序列化二叉树

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

例如，我们可以把一个只有根节点为1的二叉树序列化为"1,"，然后通过自己的函数来解析回这个二叉树



#### 62.二叉搜索树的第k个结点

**题目描述**

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）  中，按结点数值大小顺序第三小结点的值为4。

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    int index=0;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot != nullptr){
            TreeNode* node = KthNode(pRoot->left, k);
            if(node != nullptr){
                return node;
            }
            index++;
            if(index==k){
                return pRoot;
            }
            node = KthNode(pRoot->right, k);
            if(node != nullptr){
                return node;
            }
        }
        return NULL;
    }

    
};
```

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot==NULL||k<=0) return NULL;
        vector<TreeNode*> vec;
        Inorder(pRoot, vec);
        if(k>vec.size()) return NULL;
        return vec[k-1];
    }
    void Inorder(TreeNode* pRoot, vector<TreeNode*> &vec){
        if(pRoot==NULL) return;
        Inorder(pRoot->left, vec);
        vec.push_back(pRoot);
        Inorder(pRoot->right, vec);
    }

    
};
```

#### 63.数据流的中位数

**题目描述**

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。