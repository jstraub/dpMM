// =============================================================================
// == linkedList.cpp
// == --------------------------------------------------------------------------
// == A Linked List Node class
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 01-14-2008
// =============================================================================
#ifndef LINKEDLIST
#define LINKEDLIST
#include "linkedList.h"

// --------------------------------------------------------------------------
// -- linkedListNode
// --   constructor; initializes the linked list node to nothing
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>::linkedListNode() :
   data(0), prev(NULL), next(NULL)
{
}

// --------------------------------------------------------------------------
// -- linkedListNode
// --   constructor; initializes the linked list node to contain the data
// --
// --   parameters:
// --     - new_data : the data to put in the new node
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>::linkedListNode(T new_data) :
   data(new_data), prev(NULL), next(NULL)
{
}

// --------------------------------------------------------------------------
// -- getData
// --   retrieves the data at the node
// --
// --   return_value: the data of the node
// --------------------------------------------------------------------------
template <typename T>
T linkedListNode<T>::getData() const
{
   return data;
}
// --------------------------------------------------------------------------
// -- getPrev
// --   returns the pointer to the previous node
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedListNode<T>::getPrev() const
{
   return prev;
}
// --------------------------------------------------------------------------
// -- getNext
// --   returns the pointer to the previous node
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedListNode<T>::getNext() const
{
   return next;
}

template <typename T>
void linkedListNode<T>::printForward()
{
   std::cout << "Node    Data\n";

   linkedListNode<T>* temp = this;
   int count = 0;
   while (temp != NULL)
   {
      std::cout << count << "        " << temp->data << "\n";
      count++;
      temp = temp->next;
   }
}


// =============================================================================
// == linkedList.h
// == --------------------------------------------------------------------------
// == A Linked List class
// == --------------------------------------------------------------------------
// == Written by Jason Chang 01-14-2008
// =============================================================================

// --------------------------------------------------------------------------
// -- linkedList
// --   constructor; initializes the linked list to nothing
// --------------------------------------------------------------------------
template <typename T>
linkedList<T>::linkedList() :
   first(NULL), last(NULL), length(0)
{
}

// --------------------------------------------------------------------------
// -- linkedList
// --   destructor
// --------------------------------------------------------------------------
template <typename T>
linkedList<T>::~linkedList()
{
   clear();
}

// --------------------------------------------------------------------------
// -- getFirst
// --   returns the pointer to the first
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedList<T>::getFirst() const
{
   return first;
}

// --------------------------------------------------------------------------
// -- getLength
// --   returns the length of the linked list
// --------------------------------------------------------------------------
template <typename T>
int linkedList<T>::getLength() const
{
   return length;
}

// --------------------------------------------------------------------------
// -- getLength
// --   returns the length of the linked list
// --------------------------------------------------------------------------
template <typename T>
bool linkedList<T>::isempty()
{
   return length==0;
}

// --------------------------------------------------------------------------
// -- addNode
// --   adds a node to the beginning of the list
// --
// --   parameters:
// --     - new_data : the data of the new node
// --
// --   return_value: a pointer to the newly added leaf
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedList<T>::addNode(T new_data)
{
   linkedListNode<T>* newNode = new linkedListNode<T>(new_data);
   if (first != NULL)
   {
      first->prev = newNode;
      newNode->next = first;
      // newNode->prev = NULL (automatically done)
   }
   first = newNode;
   length++;

   if (length == 1 || last==NULL)
      last = first;

   return first;
}

// --------------------------------------------------------------------------
// -- addNodeEnd
// --   adds a node to the end of the list
// --
// --   parameters:
// --     - new_data : the data of the new node
// --
// --   return_value: a pointer to the newly added leaf
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedList<T>::addNodeEnd(T new_data)
{
   linkedListNode<T>* newNode = new linkedListNode<T>(new_data);
   if (last != NULL)
   {
      last->next = newNode;
      newNode->prev = last;
      // newNode->next = NULL (automatically done)
   }
   last = newNode;
   length++;

   if (length == 1 || first==NULL)
      first = last;

   return last;
}

// --------------------------------------------------------------------------
// -- addNodeUnique
// --   adds a node to the end of the list if it's unique
// --
// --   parameters:
// --     - new_data : the data of the new node
// --
// --   return_value: a pointer to the newly added leaf
// --------------------------------------------------------------------------
template <typename T>
linkedListNode<T>* linkedList<T>::addNodeUnique(T new_data)
{
   bool found = false;
   linkedListNode<T>* node = first;
   while (node!=NULL)
   {
      if (node->getData() == new_data)
         return NULL;
      node = node->getNext();
   }
   return addNodeEnd(new_data);
}

// --------------------------------------------------------------------------
// -- deleteNode
// --   deletes a node
// --
// --   parameters:
// --     - theNode : a pointer to the node to remove
// --------------------------------------------------------------------------
template <typename T>
void linkedList<T>::deleteNode(linkedListNode<T>* theNode)
{
   if (theNode == first)
      first = theNode->next;
   if (theNode == last)
      last = theNode->prev;

   if (theNode->prev != NULL)
      theNode->prev->next = theNode->next;

   if (theNode->next != NULL)
      theNode->next->prev = theNode->prev;

   length--;
   delete theNode;
}

template <typename T>
T linkedList<T>::popFirst()
{
   if (length==0 || first==NULL || last==NULL)
      mexErrMsgTxt("Popping an empty list\n");
   T returnVal = first->data;
   deleteNode(first);
   return returnVal;
}

template <typename T>
T linkedList<T>::popLast()
{
   T returnVal = last->data;
   deleteNode(last);
   return returnVal;
}

template <typename T>
void linkedList<T>::popFirst(int numToPop, linkedList<T> &poppedList)
{
   if (poppedList.length!=0 || poppedList.first!=NULL || poppedList.last!=NULL)
      mexErrMsgTxt("Non-empty popped list\n");

   if (numToPop>0)
   {
      linkedListNode<T>* poppedNode = first;
      int poppedLength = 0;
      if (poppedNode != NULL)
      {
         poppedLength++;
         for (int i=0; i<numToPop-1; i++)
         {
            if (poppedNode->next == NULL)
               break;
            poppedNode = poppedNode->next;
            poppedLength++;
         }

         poppedList.first = first;
         poppedList.last = poppedNode;
         poppedList.length = poppedLength;

         // fix this linked list
         first = poppedNode->next;
         if (first!=NULL)
            first->prev = NULL;
         else
            last = NULL;
         length -= poppedLength;
         if (length<0)
            mexErrMsgTxt("Something wrong with length in popFirst\n");

         poppedNode->next = NULL;
      }
   }
}


// --------------------------------------------------------------------------
// -- indexOf
// --   Returns the first index of data
// --------------------------------------------------------------------------
template <typename T>
int linkedList<T>::indexOf(T theData)
{
   int index = 0;
   bool found = false;
   linkedListNode<T>* node = first;
   while (node!=NULL)
   {
      if (node->getData() == theData)
      {
         found = true;
         break;
      }
      index++;
      node = node->getNext();
   }
   if (!found)
      index = -1;
   return index;
}

// --------------------------------------------------------------------------
// -- clear
// --   deletes the entire linked list
// --------------------------------------------------------------------------
template <typename T>
void linkedList<T>::clear()
{
   while (first != NULL)
   {
      linkedListNode<T>* temp = first->next;
      delete first;
      first = temp;
   }
   length = 0;
   first = NULL;
   last = NULL;
}

// --------------------------------------------------------------------------
// -- print
// --   prints the list.  used for debugging
// --------------------------------------------------------------------------
template <typename T>
void linkedList<T>::print()
{
   std::cout << "Node    Data\n";

   linkedListNode<T>* temp = first;
   int count = 0;
   while (temp != NULL)
   {
      std::cout << count << "        " << temp->data << "\n";
      count++;
      temp = temp->next;
   }
}


// --------------------------------------------------------------------------
// -- sort
// --   sorts the list.
// --------------------------------------------------------------------------
template <typename T>
void linkedList<T>::sort()
{
   if (first!=NULL)
   {
      merge_sort(first);
      last = first;
      while (last->next != NULL)
         last = last->next;
   }
}

template <typename T>
void linkedList<T>::merge_with(linkedList<T> &b)
{
   first = sorted_merge(first, b.first);
   length += b.length;
   b.first = NULL;
   b.last = NULL;
   b.length = 0;

   last = first;
   if (last!=NULL)
   {
      while (last->next != NULL)
         last = last->next;
   }
}

template <typename T>
void linkedList<T>::merge_sort(linkedListNode<T>* &a)
{
   if (a==NULL || a->next==NULL)
      return;

   linkedListNode<T> *b;
   split(a, b);

   merge_sort(a);
   merge_sort(b);

   a = sorted_merge(a,b);
}
template <typename T>
linkedListNode<T>* linkedList<T>::sorted_merge(linkedListNode<T>* a, linkedListNode<T>* b)
{
   if (a==NULL)
      return b;
   else if (b==NULL)
      return a;

   linkedListNode<T> *result, *current;
   if (a->data <= b->data)
   {
      result = a;
      a = a->next;
   }
   else
   {
      result = b;
      b = b->next;
   }
   current = result;
   current->prev = NULL;

   while (a!=NULL && b!=NULL)
   {
      if (a->data <= b->data)
      {
         current->next = a;
         a->prev = current;
         current = a;
         a = a->next;
      }
      else
      {
         current->next = b;
         b->prev = current;
         current = b;
         b = b->next;
      }
   }

   if (b!=NULL)
   {
      current->next = b;
      b->prev = current;
   }
   else if (a!=NULL)
   {
      current->next = a;
      a->prev = current;
   }

   return result;
}
template <typename T>
void linkedList<T>::split(linkedListNode<T>* a, linkedListNode<T>* &b)
{
   if (a==NULL || a->next==NULL)
      b = NULL;
   else
   {
      linkedListNode<T>* slow = a;
      linkedListNode<T>* fast = a->next;

      while (fast != NULL)
      {
         fast = fast->next;
         if (fast != NULL)
         {
            slow = slow->next;
            fast = fast->next;
         }
      }
      b = slow->next;
      b->prev = NULL;
      slow->next = NULL;
   }
}


template <typename T>
T& linkedList<T>::operator[](const int nIndex) const
{
   if (nIndex>=length)
      mexErrMsgTxt("operator[] past linked list length\n");
   linkedListNode<T>* node = first;
   for (int i=0; i<nIndex; i++)
      node = node->next;
   return node->data;
}


#endif
