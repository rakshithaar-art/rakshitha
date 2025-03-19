Implement in Java, the 0/1 Knapsack problem using 

 Greedy method. 





import java.util.*;



public class Knapsack 

{

	



	static void main(String[] args)  

	{

		Scanner s1 = new Scanner(System.in);

		System.out.println("Enter the number of objects: ");

		int n = scan.nextInt();		

                       double w[] = new double[n];

		double p[] = new double[n];

		double r[] = new double[n];

		

		System.out.println("Enter the object's weights");

		

		for (int i = 0; i<n; ++i)

			wt[i] =s1.nextDouble();

		

		System.out.println("Enter the object's profits");

		

		for (int i = 0; i<n; ++i)

			p[i] = s1.nextDouble();

		

		for (int i = 0; i<n; ++i)

			r [i] = p[i] / w[i];

		

		System.out.println("Enter the Capacity of the knapsack: ");

		

		double m= s1.nextDouble();

	

		for(int i=0;i<=n-2;i++)

			{

			for(int j=i+1;j<=n-1;j++)

				{

				if(r[i]<r[j])

				{

					double temp=r[i];

					double r[i]=r[j];

					r[j]=temp;

				}

			}

		}

knap(n,m,p,w,r);

}



 static void knap(int n, double m, double p[],double w[], double r[])

{

double profit=0.0;

double x[]=double x[n];

double rc=m;

int k=0;

while(k!=n)

{

	for(int i=0;i<n;i++)

	{

		if(r[i]==p[i]/w[i])

		{

			if(m>=w[i])

			{

				x[i]=1;

			}

			else

			{

					x[i]=rc/w[i];

				}

			profit=profit+x[i]*p[i];

			rc=rc-x[i]*w[i];

			}

		}

		k++;

	}

	system.out.println("total profit is: "+profit);

}

}





 

		{
