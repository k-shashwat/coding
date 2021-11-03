#include <bits\stdc++.h>
using namespace std;


vector<int> Solution(vector<int> A) {
    int i=0,j=0, maxlen = 0,n=A.size(),ind_i,ind_j;
    long long int sum=0,max_sum = INT_MIN;
    while(j<n){
        // cout << " i : " <<  i << " j : " << j << " maxlength : " << maxlen<< "  --  ";
        // cout << sum << " - ";
        if(A[j]<0){
            if(sum >= max_sum && j-i>maxlen){
                ind_i = i;
                ind_j = j-1;
                max_sum = sum;
                maxlen = j-i;
            }
            i = ++j;
            sum=0;
            continue;
        }
        sum = sum+A[j];
        cout << sum << endl;
        if(j==n-1) if(sum >= max_sum && j-i+1>maxlen) ind_j = j,ind_i = i;
        j++;
    }
    vector<int> ans;
    if(max_sum == INT_MIN) return ans;
    for(int i=ind_i;i<=ind_j;i++) ans.push_back(A[i]);
    return ans;
}

int main(){
    vector <int> A = {756898537, -1973594324, -2038664370, -184803526, 1424268980};
    vector<int> ans;
    ans =  Solution(A);
    for(auto g : ans) cout << g << " ";
}
